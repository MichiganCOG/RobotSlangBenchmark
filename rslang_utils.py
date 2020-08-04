from os.path import join 
import numpy as np
from constants import MazeConstants as MC
import cv2
import glob
import os
from natsort import natsorted
import socket
import tqdm
import json

def read_csv(fname):
    with open(fname, 'r') as f:
        contents = f.read().strip()
    contents = contents.split("\n")
    contents = [c.strip().split(",") for c in contents if c.strip()]
    return contents[1:]

def pose2pixel(poses, shape):
    rows, cols = shape[:2]
    pixels = np.zeros(poses[...,:2].shape)
    # X = columns, Y = inverted rows
    pixels[...,0] = (1 - poses[...,1]/MC.height)*(rows-1)
    pixels[...,1] = poses[...,0]/MC.width*(cols-1)
    return np.int32(pixels)

def pixel2pose(pixels, shape):
    rows, cols = shape[:2]
    pixels = np.float32(pixels)
    pose = np.zeros(pixels.shape)
    pose[...,1] = (rows - pixels[...,0])/rows * MC.height
    pose[...,0] = (pixels[...,1]/cols) * MC.width
    return pose

def point(pixel):
    """
    OpenCV draw functions expect points
    """
    pixel = np.squeeze(pixel).tolist()
    return (pixel[1], pixel[0])

def resize25(img):
    return cv2.resize(img, None, fx=.25, fy=.25)

def resize50(img):
    return cv2.resize(img, None, fx=.5, fy=.5)

def numify(x):
    return int("".join([_x for _x in x if _x.isdigit()]))

def to_scan_id(location, target, agent_pose, goal_pose):
    location = location.replace('/z/home/shurjo/robotslang/', '')
    location = join('mazedata', location)
    agent_pose = np.squeeze(np.round(agent_pose, 3))
    goal_pose = np.squeeze(np.round(goal_pose, 3))
    scanId = "-".join([location, target, str(agent_pose), str(goal_pose)])
    return scanId

def to_array(string):
    string = "".join([s for s in string if s not in ['[',']']])
    string = string.strip().split(' ')
    string = [s for s in string if s.strip()] 
    return np.asarray(list(map(float, string)))
            
def unit_vec(x):
    return x/np.linalg.norm(x)

def show(x, wait=-1):
    cv2.imshow('a', resize50(x))
    cv2.waitKey(wait)

def to_onehot(v, m):
    out = np.zeros(m)
    out[int(v)] = 1
    return out

def to_onehot_array(v, m):
    v = np.int32(v)
    out = np.zeros((len(v), m))
    out[np.arange(len(v)), v] = 1
    return out


def get_latest(files):
    fnames = glob.glob(files)
    fname = natsorted(fnames)[-1]
    return fname

def angle_between(v1, v2):
    v1 = unit_vec(v1)
    v2 = unit_vec(v2)
    return np.arctan2(np.cross(v1, v2), np.dot(v1, v2))

def round_angle(a, T):
    return np.round(a / T) * T

def get_hostname():
    hostname = socket.gethostname()
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu_id:
        hostname = "{}-{}".format(hostname, gpu_id)
    return hostname

def l2(p1, p2):
    return np.linalg.norm(np.squeeze(p1)[:2]-np.squeeze(p2)[:2])

def color2bgr(color):
    color = color.rgb[::-1]
    return tuple((255 * np.asarray(color)).tolist())

def softmax(vec, t=1):
    return np.exp(vec/t)/(np.exp(vec/t).sum())

def multiprocess_progress_bar(func, args):
    """https://github.com/tqdm/tqdm/issues/484#issuecomment-461998250"""
    num_proc = max(multiprocessing.cpu_count()//2, 1)
    with Pool(num_proc) as p:
        tqdm.tqdm(p.imap(func, args), total=len(args))


def makedirs(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)

def angle_range(angle):
    """
    Angle kept between [-pi, pi] range
    """
    while angle < 0:
        angle += 2*np.pi
    angle = angle % (2*np.pi)

    return angle

def angle_range_vector(angles):
    """
    Angle kept between [0, 2*pi] range
    """
    # Negative angles to positve angles
    angles[angles<0] += 2*np.pi
    # Keep to [0, 2*pi] range
    angles = angles % (2*np.pi)

    return angles

def load_data(splits=["train", "val_seen", "test"]):
    data = []
    for split in splits:
        with open('data/{}.json'.format(split)) as f:
            data += json.load(f)
    return data
