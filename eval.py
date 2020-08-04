''' Evaluation of agent trajectories '''

import json
import os
import sys
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from utils import load_datasets
from agent import BaseAgent, StopAgent, RandomAgent, ShortestAgent

from rslang_simulator import RobotSlangSimulator
from constants import EpisodeConstants as EC

def path_len(traj):
    return traj[-1]

def to_meters(path):
    if len(path) <= 1:
        return 0
    else:
        path = np.asarray(path)[:,:2]
        return np.linalg.norm(path[1:] - path[:-1], axis=1).sum()

class Evaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, path_type='planner_path'):
        self.error_margin = 1. #EC.dist_threshold  # cm
        self.splits = splits
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        for item in load_datasets(splits):
            self.gt[item['inst_idx']] = item
            self.instr_ids.append(item['inst_idx'])
            self.scans.append(item['scan'])

        self.instr_ids = set(self.instr_ids)
        self.distances = {}
        self.path_type = path_type

    
    def _get_nearest(self, distances, goal_id, path):
        dists = [distances[(p, goal_id)][-1] for p in path]
        return path[np.argmin(dists)]

    def _score_item(self, instr_id, path, scan):
        ''' Calculate error based on the final position in trajectory, and also 
            the closest position (oracle stopping rule). '''
        
        # shortest path (load it one at a time)
        env = RobotSlangSimulator(scan)
        planner = env.planner
        # Planner to get topological distanced

        gt    = self.gt[int(instr_id)]
        start = env.agent.pose[:,:2][0]  
        goal  = env.goal.pose[:,:2][0]
        final_position = path[-1]

        # Distance from final position to goal
        top_final_goal = planner.get_top_dist(final_position, goal)
        euc_final_goal = planner.get_euc_dist(final_position, goal)
        
        # topolocal distance from oracle position to goal
        top_oracle_goal = np.min([planner.get_top_dist(p, goal) for p in path])
        euc_oracle_goal = np.min([planner.get_euc_dist(p, goal) for p in path])


        # Store the metrics
        self.scores['top_final_goal'].append(top_final_goal)
        self.scores['top_oracle_goal'].append(top_oracle_goal)
        self.scores['euc_final_goal'].append(euc_final_goal)
        self.scores['euc_oracle_goal'].append(euc_oracle_goal)


    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        with open(output_file) as f:
            for item in json.load(f):
                # Check against expected ids
                if item['inst_idx'] in instr_ids:
                    instr_ids.remove(item['inst_idx'])
                    self._score_item(item['inst_idx'], item['trajectory'], item['scan'])
        
        for key in self.scores:
            self.scores[key] = np.average(self.scores[key])

        return dict(self.scores)


from rslang_utils import get_hostname

RESULT_DIR = 'tasks/NDH/results/{}/'.format(get_hostname())


def eval_simple_agents():
    # path_type = 'planner_path'
    # path_type = 'player_path'
    path_type = 'trusted_path'

    ''' Run simple baselines on each split. '''
    for split in ['train', 'val_seen', 'val_unseen']:
        env = R2RBatch(None, batch_size=1, splits=[split], path_type=path_type)
        ev = Evaluation([split], path_type=path_type)

        for agent_type in ['Stop', 'Shortest', 'Random']:
            outfile = '%s%s_%s_agent.json' % (RESULT_DIR, split, agent_type.lower())
            agent = BaseAgent.get_agent(agent_type)(env, outfile)
            agent.test()
            agent.write_results()
            score_summary, _ = ev.score(outfile)
            print('\n%s' % agent_type)
            pp.pprint(score_summary)


def eval_seq2seq():
    ''' Eval sequence to sequence models on val splits (iteration selected from training error) '''
    outfiles = [
        RESULT_DIR + 'seq2seq_teacher_imagenet_%s_iter_5000.json',
        RESULT_DIR + 'seq2seq_sample_imagenet_%s_iter_20000.json'
    ]
    for outfile in outfiles:
        for split in ['val_seen', 'val_unseen']:
            ev = Evaluation([split])
            score_summary, _ = ev.score(outfile % split)
            print('\n%s' % outfile)
            pp.pprint(score_summary)


if __name__ == '__main__':

    eval_simple_agents()
    #eval_seq2seq()
