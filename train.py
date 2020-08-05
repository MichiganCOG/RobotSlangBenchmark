import argparse 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

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


# VOCAB
TRAIN_VOCAB    = '{}/train_vocab.txt'.format(Files.data)
TRAINVAL_VOCAB = '{}/trainval_vocab.txt'.format(Files.data)

# Training settings.
agent_type = 'seq2seq'

# Fixed params from MP.
batch_size = 100
word_embedding_size = 128
action_embedding_size = 8
target_embedding_size = 2
hidden_size = 128
bidirectional = False
dropout_ratio = 0.5
weight_decay = 0.0005
feature_size = 144
MAX_EPISODE_LEN = EC.max_len 
MAX_INPUT_LENGTH = 100  

# Original defaults
RESULT_DIR     = None 
SNAPSHOT_DIR   = None 
PLOT_DIR       = None 
LEARNING_RATE  = None 

def set_defaults(fold, feedback, blind, learning_rate=0.0001):
    global RESULT_DIR, SNAPSHOT_DIR, PLOT_DIR, LEARNING_RATE
    par_folder     = '{}_feedback-{}_blind-{}_lr-{:f}'.format(
                                fold, feedback, blind, learning_rate)
    RESULT_DIR     = '{}/{}/'.format(Files.results  , par_folder)
    SNAPSHOT_DIR   = '{}/{}/'.format(Files.snapshots, par_folder)
    PLOT_DIR       = '{}/{}/'.format(Files.plots    , par_folder)
    for dirs in [RESULT_DIR, SNAPSHOT_DIR, PLOT_DIR]:
        if not os.path.isdir(dirs):
            os.makedirs(dirs)
    LEARNING_RATE = learning_rate

def train(train_env, encoder, decoder, n_iters, path_type, history, feedback_method, max_episode_len, max_input_length, model_prefix,
    log_every=50, val_envs=None):
    ''' Train on training set, validating on both seen and unseen. '''
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

def setup():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # Check for vocabs
    if not os.path.exists(TRAIN_VOCAB):
        write_vocab(build_vocab(splits=['train']), TRAIN_VOCAB)
    if not os.path.exists(TRAINVAL_VOCAB):
        write_vocab(build_vocab(splits=['train', 'val_seen']), TRAINVAL_VOCAB)



def train_val(path_type, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind):
    ''' Train on the training set, and validate on seen and unseen splits. '''
  
    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=max_input_length)
    train_env = R2RBatch(batch_size=batch_size, splits=['train'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(batch_size=batch_size, splits=[split], 
                tokenizer=tok, path_type=path_type, history=history, blind=blind),
                Evaluation([split], path_type=path_type)) for split in ['val_seen']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx, 
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio, feature_size).cuda()
    train(train_env, encoder, decoder, n_iters,
          path_type, history, feedback_method, max_episode_len, max_input_length, model_prefix, val_envs=val_envs)


def train_test(path_type, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind):
    ''' Train on the training set, and validate on the test split. '''

    setup()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(TRAINVAL_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=MAX_INPUT_LENGTH)
    train_env = R2RBatch(batch_size=batch_size, splits=['train', 'val_seen'], tokenizer=tok,
                         path_type=path_type, history=history, blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(batch_size=batch_size, splits=[split], 
                tokenizer=tok, path_type=path_type, history=history, blind=blind),
                Evaluation([split], path_type=path_type)) for split in ['test']}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx, 
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio, feature_size).cuda()
    train(train_env, encoder, decoder, n_iters,
          path_type, history, feedback_method, max_episode_len, max_input_length, model_prefix, val_envs=val_envs)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback', type=str, required=True,
                        help='teacher or sample')
    parser.add_argument('--eval_type', type=str, required=True,
                        help='val or test')
    parser.add_argument('--blind', type=str, required=False, default='',
                        help='language, vision')
    parser.add_argument('--lr', type=float, required=False, default=0.0001)
    
    args = parser.parse_args()

    assert args.feedback in ['sample', 'teacher']
    assert args.eval_type in ['val', 'test']

    # Set the default folder locations
    set_defaults(args.eval_type, args.feedback, args.blind, args.lr)

    blind = args.blind

    # Set default args.
    path_type = 'path-planner'
    max_episode_len = MAX_EPISODE_LEN
    history = 'all' 
    max_input_length = MAX_INPUT_LENGTH
    
    feedback_method = args.feedback
    n_iters = 5000 if feedback_method == 'teacher' else 20000

    # Model prefix to uniquely id this instance.
    model_prefix = '%s-seq2seq-%d-%s' % (args.eval_type, max_episode_len, feedback_method)
    if blind:
        model_prefix += '-blind={}'.format(blind)

    if args.eval_type == 'val':
        train_val(path_type, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind)
    else:
        train_test(path_type, max_episode_len, history, max_input_length, feedback_method, n_iters, model_prefix, blind)

    # test_submission(path_type, max_episode_len, history, MAX_INPUT_LENGTH, feedback_method, n_iters, model_prefix, blind)
