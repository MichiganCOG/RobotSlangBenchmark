''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import csv
import numpy as np
import math
import base64
import json
import random
import networkx as nx

from utils import load_datasets
from rslang_simulator import RobotSlangSimulator
import time 

csv.field_size_limit(sys.maxsize)

class EnvBatch():
    ''' A simple wrapper for a batch of RobotSlang environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size=100, blind=False):
        self.envs = dict() 

    def newEpisodes(self, scanIds):
        self.scanIds = scanIds
        for sc in scanIds:
            if sc not in self.envs:
                self.envs[sc] = RobotSlangSimulator(sc)
            else:
                self.envs[sc].reset()

    def getStates(self):
        return [self.envs[sc] for sc in self.scanIds]

    def makeActions(self, actions):
        # Save ordering as scan ids
        for i,sc in enumerate(self.scanIds):
            self.envs[sc].makeActions(actions[i])




class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 path_type='planner_path', history='target', blind=False):
        self.env = EnvBatch(batch_size=batch_size, blind=blind)
        self.data = []
        self.scans = []
        for item in load_datasets(splits):

            # For every dialog history, stitch together a single instruction string.
            self.scans.append(item['scan'])
            new_item = dict(item)
            new_item['inst_idx'] = item['inst_idx']
            
            # history == 'all':
            dia_inst = ''
            sentences = []
            seps = []
            for turn in item['dialog_history']:
                sentences.append(turn['message'])
                sep = '<NAV>' if turn['role'] == 'navigator' else '<ORA>'
                seps.append(sep)
                dia_inst += sep + ' ' + turn['message'] + ' '
            sentences.append(item['target'])
            seps.append('<TAR>')
            dia_inst += '<TAR> ' + item['target']
            new_item['instructions'] = dia_inst
            
            if tokenizer:
                dia_enc = tokenizer.encode_sentence(sentences, seps=seps)
                new_item['instr_encoding'] = dia_enc
            
            self.data.append(new_item)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self.path_type = path_type
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

        self.timer = time.time()
        self.blind = blind


    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
            self.ix = self.batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += self.batch_size
        #print(self.ix+self.batch_size, len(self.data),',', time.time()-self.timer)
        #self.timer = time.time()
        self.batch = batch

    def reset_epoch(self):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        self.ix = 0

    def _get_obs(self):
        obs = []
        for i,state in enumerate(self.env.getStates()):
            item = self.batch[i]
            obs.append({
                'inst_idx': item['inst_idx'],
                'target': state.target, 
                'scan': state.scanId,
                'agent': state.agent.pose, 
                'feature': state.get_obs(),
                'instructions': item['instructions'],
                'teacher': state.next_shortest_path_action(),
                'collision': state.check_collision(),
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            
            if self.blind == 'vision':
                obs[-1]['feature'] *= 0
        
        return obs

    def reset(self):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch()
        scanIds = [item['scan'] for item in self.batch]
        self.env.newEpisodes(scanIds)
        return self._get_obs()   

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()


