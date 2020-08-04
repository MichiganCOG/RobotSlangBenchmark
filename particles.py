import numpy as np
from rslang_utils import angle_range_vector
from filterpy.monte_carlo import resampling
import random
from constants import MazeConstants as MC
from rslang_utils import pose2pixel,point 
import cv2

WHITE = (255, 255, 255)
RED   = (  0,   0, 255)


class Particles:
    """The particle's pose, weights, etc are stored in a giant matrix 
    to allow for easy parallelization.
    """
    def __init__(self, num_particles, mapfile=None, pose=None, r_mean=0., r_std=0.01, angle_mean=0., angle_std=np.radians(30), border_width=0.1, x_range=MC.width, y_range=MC.height):

        # Constants
        self.num_particles = num_particles 
        self.mapfile = mapfile
        self.x_range, self.y_range = x_range, y_range
        self.border_width = border_width
        self.r_mean, self.r_std = r_mean, r_std
        self.angle_mean, self.angle_std = angle_mean, angle_std

        # Maze limits
        self.xmin, self.xmax = self.border_width, self.x_range-self.border_width 
        self.ymin, self.ymax = self.border_width, self.y_range-self.border_width
        
        # Initialize the pose matrix
        self.pose = self.sample_particles(self.num_particles)
        self.weights = np.ones(num_particles)/num_particles

    def sample_particles(self, num_particles):
        ret = False
        pose = np.zeros((num_particles, 3))
        #while not ret:
        # Sample uniformly from whole map
        pose[...,0] = np.random.uniform(self.xmin, self.xmax, num_particles) 
        pose[...,1] = np.random.uniform(self.ymin, self.ymax, num_particles)
        pose[...,2] = np.random.uniform(        0,   2*np.pi, num_particles)
        #ret = self.no_collision_dectection(pose)
        return pose
    
    def random_motion(self):
        # Move the particles with the motion model
        dr = np.random.normal(self.r_mean, self.r_std, self.num_particles)
        dangle = np.random.normal(self.angle_mean, self.angle_std, self.num_particles)
        self.move(dr, dangle)

    def move(self, dr, dangle):
        self.pose = self.next_pose(dr, dangle) 
        return self.pose 
    
    def next_pose(self, dr, dangle):
        new_pose = self.pose.copy()
        new_pose[...,0] += dr * np.cos(new_pose[...,2])
        new_pose[...,1] += dr * np.sin(new_pose[...,2])
        new_pose[...,2] += dangle
        new_pose[...,2] = angle_range_vector(new_pose[...,2])
        new_pose = self.within_range(new_pose)
        return new_pose
    
    def within_range(self, pose):
        pose[...,0][pose[...,0]<self.xmin] = self.xmin
        pose[...,0][pose[...,0]>self.xmax] = self.xmax
        pose[...,1][pose[...,1]<self.ymin] = self.ymin
        pose[...,1][pose[...,1]>self.ymax] = self.ymax
        pose[...,2] = angle_range_vector(pose[...,2])
        return pose

    def resample(self):
        new_ndxs = resampling.systematic_resample(self.weights)
        self.pose    = self.pose[new_ndxs].copy()
        self.weights = self.weights[new_ndxs].copy()
    
    def __len__(self):
        return self.num_particles
    
    def __repr__(self):
        return self.pose
