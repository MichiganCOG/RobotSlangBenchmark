from rslang_simulator import RobotSlangSimulator
from utils import load_datasets 
from rslang_utils import show
import os
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import glob
import numpy as np

if not os.path.isdir('videos'):
    os.makedirs('videos')

def print_dialog(data):
    for d in data['dialog_history']:
        role = 'Commander' if d['role'] == 'oracle' else 'Driver'
        print('%9s: %s' %(role, d['message']))
    print()

def make_video(data):
    env = RobotSlangSimulator(data['scan'])
    # Video creation 
    fourcc = VideoWriter_fourcc(*'MP42')
    vfile  = 'videos/{}.avi'.format(str(env))
    fps    = 15
    height, width, _ = env.display(draw_measurements=True).shape
    video  = VideoWriter(vfile, fourcc, float(fps), (width, height))
    for display in env.shortest_agent_images():
        video.write(display)
    video.release()
        


if __name__ == "__main__":
    import multiprocessing
    from multiprocessing import Pool
    import tqdm
    
    print("Making trial videos")
    def cache_shortest_path(mapfile):
        planner = FloydWarshallRobotslang(mapfile)
    num_proc = max(multiprocessing.cpu_count()//2, 1)
    with Pool(num_proc) as p:
        list(tqdm.tqdm(p.imap(make_video, data), total=len(data)))
    print()
