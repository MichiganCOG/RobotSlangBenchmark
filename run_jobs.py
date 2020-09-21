import os
import subprocess
import numpy as np
import sys
import time
from time import sleep
from datetime import datetime

def make_args(**kwargs):
    new_kwargs = ['--{} {}'.format(k, v) for k,v in kwargs.items()]
    return " ".join(new_kwargs)

COMMAND = """sbatch --export=ARGS="{}" --job-name={} run_job.sb"""
PYTHON = """python train_new.py --feedback={} --eval_type={} --blind={} --lr={} --seed {} --hidden_size {} --word_embedding_size {}"""
def main():
    count = 0
    lrs = dict(teacher=0.0001, sample=0.001)
    not_confirmed = True 
    
    for seed in range(3):
        for fold in ['val', 'test']:
            for feedback in ['teacher', 'sample']:
                for blind in ['', 'language', 'vision']:
                    for size in [64]:
                        lr = lrs[feedback]
                        jobname = "-".join([feedback, fold, blind, str(lr)])
                        command = PYTHON.format(feedback, fold, blind, lr, seed, size, size)
                        command = COMMAND.format(command, jobname)
                        print(command)
                        if not_confirmed:
                            x = input("Are you sure want to run jobs on the cluster")
                            if x == 'y':
                                not_confirmed = False
                        if not not_confirmed: 
                            import ipdb; ipdb.set_trace()
                            subprocess.call(command, shell=True)
                        count += 1

    return count    
                    

if __name__ == "__main__":
    print(main())
