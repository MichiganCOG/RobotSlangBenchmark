import click 
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

# Ray tune hyperparameter optimization
from functools import partial
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.schedulers import ASHAScheduler

from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch
from wandb.sweeps.config.hyperopt import hp


def output_location(eval_type, feedback, blind, lr, hidden_size, word_embedding_size, seed):
    par_eval_typeer     = '{}_feedback-{}_blind-{}_lr-{:f}_hs-{}_we-{}'.format(
                    eval_type, feedback, blind, lr, hidden_size, word_embedding_size)
    result_dir     = '{}/{}/seed-{}/'.format(Files.results  , par_eval_typeer, seed)
    snapshot_dir   = '{}/{}/seed-{}/'.format(Files.snapshots, par_eval_typeer, seed)
    plot_dir       = '{}/{}/seed-{}/'.format(Files.plots    , par_eval_typeer, seed)
    for dirs in [result_dir, snapshot_dir, plot_dir]:
        if not os.path.isdir(dirs):
            os.makedirs(dirs)
    return result_dir, snapshot_dir, plot_dir




@wandb_mixin
def train(config):
    ''' Train on training set, validating on both seen and unseen. '''
    
    #############################################################
    # FIXME
    #############################################################
    keys = """eval_type, seed, max_episode_len, max_input_length, feedback,
    n_iters, prefix, blind, debug, train_vocab, trainval_vocab, batch_size,
    action_embedding_size, target_embedding_size, bidirectional,
    dropout_ratio, weight_decay, feature_size, hidden_size,
    word_embedding_size, lr, result_dir, snapshot_dir, plot_dir, train_splits,
    test_splits""".split(',')
    keys = [k.strip() for k in keys if k.strip()]
    eval_type, seed, max_episode_len, max_input_length, feedback, n_iters, prefix, blind, debug, train_vocab, trainval_vocab, batch_size, action_embedding_size, target_embedding_size, bidirectional,dropout_ratio, weight_decay, feature_size, hidden_size, word_embedding_size, lr, result_dir, snapshot_dir, plot_dir, train_splits, test_splits = [config[k] for k in keys]
    log_every=50 
    #############################################################
    
    setup(seed, train_vocab, trainval_vocab)
    
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(train_vocab if eval_type=='val' else trainval_vocab)
    tok = Tokenizer(vocab=vocab, encoding_length=max_input_length)
    train_env = R2RBatch(batch_size=batch_size, splits=train_splits, tokenizer=tok, seed=seed, blind=blind)

    # Creat validation environments
    val_envs = {split: (R2RBatch(batch_size=batch_size, splits=[split], 
                tokenizer=tok, seed=seed, blind=blind),
                Evaluation([split], seed=seed)) for split in test_splits}

    # Build models and train
    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    encoder = EncoderLSTM(len(vocab), word_embedding_size, enc_hidden_size, padding_idx, 
                  dropout_ratio, bidirectional=bidirectional).cuda()
    decoder = AttnDecoderLSTM(Seq2SeqAgent.n_inputs(), Seq2SeqAgent.n_outputs(),
                  action_embedding_size, hidden_size, dropout_ratio, feature_size).cuda()
    if debug: 
        print("Training in debug mode")
        log_every = 1

    if val_envs is None:
        val_envs = {}

    print('Training with %s feedback' % (feedback))
    agent = Seq2SeqAgent(train_env, "", encoder, decoder, max_episode_len, blind=blind, debug=debug)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay) 

    data_log = defaultdict(list)
    start = time.time()
   
    for idx in range(0, n_iters, log_every):

        interval = min(log_every,n_iters-idx)
        iter = idx + interval
        data_log['iteration'].append(iter)

        # Train for log_every interval
        agent.train(encoder_optimizer, decoder_optimizer, interval, feedback=feedback)
        train_losses = np.array(agent.losses)
        assert len(train_losses) == interval
        train_loss_avg = np.average(train_losses)
        data_log['train loss'].append(train_loss_avg)
        loss_str = 'train loss: %.4f' % train_loss_avg

        # Run validation
        for env_name, (env, evaluator) in val_envs.items():
            agent.env = env
            agent.results_path = '%s%s_%s_iter_%d.json' % (result_dir, prefix, env_name, iter)
            # Get validation loss under the same conditions as training
            agent.test(use_dropout=True, feedback=feedback, allow_cheat=True)
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
        df_path = '%s%s-log.csv' % (plot_dir, prefix)
        df.to_csv(df_path)
        
        split_string = "-".join(train_env.splits)
        enc_path = '%s%s_%s_enc_iter_%d' % (snapshot_dir, prefix, split_string, iter)
        dec_path = '%s%s_%s_dec_iter_%d' % (snapshot_dir, prefix, split_string, iter)
        agent.save(enc_path, dec_path)
        
        # Log data to wandb for visualization
        wandb.log(last_entry(data_log, eval_type), step=idx)
        tune.report(**last_entry(data_log, eval_type))

def last_entry(data_log, eval_type):
    # Separate metrics and losses in to different sections in wandb
    out = {k.replace(' ', '_'): v[-1] for k,v in data_log.items()}
    del out['iteration']
    new_out = {}
    for k,v in out.items():
        new_out['{}-{}/{}'.format(eval_type, 'loss' if 'loss' in k else 'metrics', k)] = v
    return new_out

def setup(seed, train_vocab, trainval_vocab):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    # Check for vocabs
    if not os.path.exists(train_vocab):
        write_vocab(build_vocab(splits=['train']), train_vocab)
    if not os.path.exists(trainval_vocab):
        write_vocab(build_vocab(splits=['train', 'val_seen']), trainval_vocab)


def train_val(*args, **kwargs):
    train_all(*args, **kwargs, train_splits=['train'], test_splits=['val_seen'])

def train_test(*args, **kwargs):
    train_all(*args, **kwargs, train_splits=['train', 'val_seen'], test_splits=['test'])



@click.command()
@click.option('--feedback', default='sample', type=str, help='teacher or sample')
@click.option('--eval_type', default='val', type=str, help='val or test')
@click.option('--blind', type=str,  default='', help='language, vision')
@click.option('--lr', type=float,  default=0.0001)
@click.option('--seed', type=int,  default=1)
@click.option('--debug', type=bool, default=False, help='Run in debug mode')
@click.option('--hidden_size', type=int,  default=64)
@click.option('--word_embedding_size', type=int,  default=64)
@click.option('--n_iters', type=int,  default=2000)
@click.option('--max_episode_len', type=int,  default=EC.max_len)
@click.option('--max_input_length', type=int, default=100)
@click.option('--batch_size', default=100)
@click.option('--action_embedding_size', default=4)
@click.option('--target_embedding_size', default=2)
@click.option('--bidirectional', default=False)
@click.option('--dropout_ratio', default=0.5)
@click.option('--weight_decay',  default=0.0005)
@click.option('--feature_size',  default=144)
@click.option('--project', default='RobotSlangBenchmarkRayTune')
def main(feedback, eval_type, blind, lr, seed, debug, hidden_size, word_embedding_size, n_iters, max_episode_len, max_input_length, batch_size, action_embedding_size, target_embedding_size, bidirectional, dropout_ratio, weight_decay, feature_size, project):
    
    # Vocabulary locations
    train_vocab    = '{}/train_vocab.txt'.format(Files.data)
    trainval_vocab = '{}/trainval_vocab.txt'.format(Files.data)

    # Model prefix to uniquely id this instance.
    prefix = 'feedback_%s-hs_%d-we_%d' % (feedback, hidden_size, word_embedding_size)
    if blind: prefix += '-blind_{}'.format(blind)
    if debug: prefix += '-debug={}'.format(debug)
    
    # Store results in pre-determined directories
    result_dir, snapshot_dir, plot_dir = output_location(eval_type, feedback, blind, lr, hidden_size, word_embedding_size, seed)

    # Initialize wandb for experiment tracking
    if debug: project = 'DEBUG-' + project
    #wandb.init(project=project, name=prefix, config=locals())
    
    # Test on validation or test folds based on data
    train_func  = train_val if eval_type == 'val' else train_test
    train_func(eval_type, seed, max_episode_len, max_input_length, feedback,
               n_iters, prefix, blind, debug, train_vocab, trainval_vocab,
               batch_size, action_embedding_size, target_embedding_size,
               bidirectional, dropout_ratio, weight_decay, feature_size,
               hidden_size, word_embedding_size, lr, result_dir, snapshot_dir, plot_dir)

    # test_submission(seed, max_episode_len, max_input_length, feedback, n_iters, prefix, blind)


def train_all(eval_type, seed, max_episode_len, max_input_length, feedback,
              n_iters, prefix, blind, debug, train_vocab, trainval_vocab, batch_size,
              action_embedding_size, target_embedding_size, bidirectional,
              dropout_ratio, weight_decay, feature_size, hidden_size,
              word_embedding_size, lr, result_dir, snapshot_dir, plot_dir, train_splits,
              test_splits):
    ''' Train on the training set, and validate on the test split. '''
    config = locals().copy()
    config.update({
        # Hyper-parameters sweep
        'batch_size'         : tune.grid_search([25, 50, 100]),
        'hidden_size'        : tune.grid_search([32, 64, 128]),
        'word_embedding_size': tune.grid_search([32, 64, 128]),
        'lr'                 : tune.grid_search([1e-4, 1e-3, 1e-2]),
        'max_input_length'   : tune.grid_search([25, 50, 100]),
        #"local_dir"          : "/z/home/shurjo/reborn/RobotSlangBenchmark",
        'scheduler' : ASHAScheduler(metric="val-metrics/val_seen_top_final_goal", mode="min", grace_period=1),
        # wandb configuration
        "wandb": {"project": "RobotSlangBenchmarkRayTune"},
        "num_gpu": 10,
        }),
    tune.run(train, config=config, loggers=DEFAULT_LOGGERS + (WandbLogger,), resources_per_trial={"gpu": 1, "cpu": 7})
    
    
    



if __name__ == "__main__":
    main()


