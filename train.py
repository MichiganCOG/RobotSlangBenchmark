import argparse 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import random

import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_vocab,write_vocab,build_vocab,Tokenizer,padding_idx,timeSince
from env import R2RBatch
from model import EncoderLSTM, AttnDecoderLSTM
from agent import Seq2SeqAgent
from eval import Evaluation

from constants import EpisodeConstants as EC, Files 

import wandb

# VOCAB
TRAIN_VOCAB    = '{}/train_vocab.txt'.format(Files.data)
TRAINVAL_VOCAB = '{}/trainval_vocab.txt'.format(Files.data)

# Training settings.
agent_type = 'seq2seq'

# Fixed params from MP.
word_embedding_size = 64
hidden_size         = 64

batch_size = 100
action_embedding_size = 4
target_embedding_size = 2
bidirectional = False
dropout_ratio = 0.5
weight_decay = 0.0005
feature_size = 144
MAX_EPISODE_LEN = EC.max_len 
MAX_INPUT_LENGTH = 100  

# Original defaultu
RESULT_DIR     = None 
SNAPSHOT_DIR   = None 
PLOT_DIR       = None 
LEARNING_RATE  = None 

def set_defaults(args, model_prefix, fold, feedback, blind, learning_rate=0.0001):
    global RESULT_DIR, SNAPSHOT_DIR, PLOT_DIR, LEARNING_RATE
    global LEARNING_RATE, hidden_size, word_embedding_size
    par_folder     = '{}_feedback-{}_blind-{}_lr-{:f}_hs-{}_we-{}'.format(
                    fold, feedback, blind, learning_rate, hidden_size, word_embedding_size)
    RESULT_DIR     = '{}/{}/seed-{}/'.format(Files.results  , par_folder, args.seed)
    SNAPSHOT_DIR   = '{}/{}/seed-{}/'.format(Files.snapshots, par_folder, args.seed)
    PLOT_DIR       = '{}/{}/seed-{}/'.format(Files.plots    , par_folder, args.seed)
    for dirs in [RESULT_DIR, SNAPSHOT_DIR, PLOT_DIR]:
        if not os.path.isdir(dirs):
            os.makedirs(dirs)
    
    LEARNING_RATE       = learning_rate
    hidden_size         = args.hidden_size
    word_embedding_size = args.word_embedding_size
    
    
    # Update config with local and global hyper-parameters 
    # FIXME Globals are bad - move to a argparse/click based parameter input system
    params = dict(args._get_kwargs())
    params.update(locals())
    params.update(globals())
    value_param = lambda k, v: isinstance(v, (float, bool, str, int)) and '__' not in k
    params = {k:v for k,v in params.items() if value_param(k,v)}
    
    # Initialize wandb for experiment tracking
    project = 'RobotSlangBenchmarkEND-2'
    if args.debug:
        project = 'DEBUG-' + project
    wandb.init(project=project, name=model_prefix)
    # Register all params for grouping purposes 
    wandb.config.update(params)



def train(eval_type, train_env, encoder, decoder, n_iters, seed, history, feedback_method, max_episode_len, max_input_length, model_prefix,
    log_every=50, val_envs=None, debug=False):
    ''' Train on training set, validating on both seen and unseen. '''
    
    if debug: 
        print("Training in debug mode")
        log_every = 1

    if val_envs is None:
        val_envs = {}

    if agent_type == 'seq2seq':
        agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len, blind=blind)
    else:
        sys.exit("Unrecognized agent_type '%s'" % agent_type)
    print('Training a %s agent with %s feedback' % (agent_type, feedback_method))
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=LEARNING_RATE, weight_decay=weight_decay) 

    data_log = defaultdict(list)
    start = time.time()
   
    for idx in range(0, n_iters, log_every):

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback_method)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (RESULT_DIR, model_prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback_method, allow_cheat=True)
            val_losses = np.array(agent.losses)
            val_loss_avg = np.average(val_losses)
            data_log['%s loss' % env_name].append(val_loss_avg)
            # Get validation distance from goal under test evaluation conditions
            agent.test(use_dropout=False, feedback='argmax')
            agent.write_results()
            score_summary = evaluator.score(agent.results_path)
            loss_str += ', %s loss: %.4f' % (env_name, val_loss_avg)
            for metric, val in score_summary.items():
                data_log['%s %s' % (env_name, metric)].append(val)
                loss_str += ', %s: %.3f' % (metric, val)


        agent.env = train_env

        print(('%s (%d %d%%) %s' % (timeSince(start, float(iter)/n_iters),
                                             iter, float(iter)/n_iters*100, loss_str)))
        df = pd.DataFrame(data_log)
        df.set_index('iteration')
        df_path = '%s%s-log.csv' % (PLOT_DIR, model_prefix)
        df.to_csv(df_path)
        
        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (SNAPSHOT_DIR, model_prefix, split_string, iter)
        agent.save(enc_path, dec_path)
        
        # Log data to wandb for visualization
        wandb.log(last_entry(data_log, eval_type), step=idx)

def last_entry(data_log, eval_type):
    # Separate metrics and losses in to different sections in wandb
    out = {k.replace(' ', '_'): v[-1] for k,v in data_log.items()}
    del out['iteration']
    new_out = {}
    for k,v in out.items():
        new_out['{}-{}/{}'.format(eval_type, 'loss' if 'loss' in k else 'metrics', k)] = v
    return new_out

def setup(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen']), TRAINVAL_VOCAB)



def train_val(eval_type, seed, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind, debug):
    ''' Train on the training set, and validate on seen and unseen splits. '''
  
    setup(seed)
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=max_input_length)
    train_env = R2RBatch(batch_size=batch_size, splits=['train'], tokenizer=tok,
                         seed=seed, history=history, blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(batch_size=batch_size, splits=[split], 
                tokenizer=tok, seed=seed, history=history, blind=blind),
                Evaluation([split], seed=seed)) for split in ['val_seen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx, 
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio, feature_size).cuda()
    train(eval_type, train_env, encoder, decoder, n_iters,
          seed, history, feedback_method, max_episode_len, max_input_length, model_prefix, val_envs=val_envs, debug=debug)


def train_test(eval_type, seed, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind, debug):
    ''' Train on the training set, and validate on the test split. '''

    setup(seed)
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(batch_size=batch_size, splits=['train', 'val_seen'], tokenizer=tok,
                         seed=seed, history=history, blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(batch_size=batch_size, splits=[split], 
                tokenizer=tok, seed=seed, history=history, blind=blind),
                Evaluation([split], seed=seed)) for split in ['test']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx, 
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio, feature_size).cuda()
    train(eval_type, train_env, encoder, decoder, n_iters,
          seed, history, feedback_method, max_episode_len, max_input_length, model_prefix, val_envs=val_envs, debug=debug)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback', type=str, required=True,
                        help='teacher or sample')
    parser.add_argument('--eval_type', type=str, required=True,
                        help='val or test')
    parser.add_argument('--blind', type=str, required=False, default='',
                        help='language, vision')
    parser.add_argument('--lr', type=float, required=False, default=0.0001)
    parser.add_argument('--seed', type=int, required=False, default=1)
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--hidden_size', type=int, required=False, default=64)
    parser.add_argument('--word_embedding_size', type=int, required=False, default=64)
    
    args = parser.parse_args()

    assert args.feedback in ['sample', 'teacher']
    assert args.eval_type in ['val', 'test']

    # Set the default folder locations
    feedback_method = args.feedback
    #n_iters = 5000 if feedback_method == 'teacher' else 20000
    n_iters = 2000
    args.n_iters = n_iters
    max_episode_len = MAX_EPISODE_LEN
    
    # Set default args.
    blind = args.blind
    seed = args.seed
    history = 'all' 
    max_input_length = MAX_INPUT_LENGTH
    debug = args.debug
    
    # Model prefix to uniquely id this instance.
    model_prefix = 'feedback_%s-hs_%d-we_%d' % (feedback_method, args.hidden_size, args.word_embedding_size)
    if blind:
        model_prefix += '-blind_{}'.format(blind)
    if debug:
        model_prefix += '-debug={}'.format(debug)

    # Set global output variables
    set_defaults(args, model_prefix, args.eval_type, args.feedback, args.blind, args.lr)

    
    if args.eval_type == 'val':
        train_val(args.eval_type, args.seed, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind, debug)
    else:
        train_test(args.eval_type, seed, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind, debug)

    # test_submission(seed, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)
