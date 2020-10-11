"""
Python program to run particle filter based localization 
on the robotslang dataset. The system utlizes the front
facing visual inputs from the system to localize the 
robot's position within the maze from the top down 
perspective.
Visual localization of robot using a particle filtering approach

"""
#import sys; import os; sys.path.append(os.getcwd())
import cv2
import numpy as np
import os
import glob
import time 
import random 
from particles import Particles
import socket
import pickle
from rslang_utils import pose2pixel, point, show
from tqdm import tqdm

from constants import MazeConstants as MC, Action
from rslang_utils import to_onehot_array, color2bgr, softmax, angle_range_vector
from labels import LABELS, label2color
from histogram_features import HistogramFeatures

from scipy.signal import correlate
from colour import Color
import click
from rslang_simulator import RobotSlangSimulator
from utils import load_datasets


class ParticleFilter:
    """
    Class representation of visual particle filter.
    """
    def __init__(self, env, num_particles, particle_angle_range, num_rays,
                 r_mean, r_std, ang_mean, ang_std, visualize, background, save_to,
                 row_min, row_max, n_angle_bins, front_angle_range):
        """Instantiates the visual particle filter with initial constraints.
        Args:
            run (dataset object): Provides access to an experiment
            experiment  Location of the experiment
            mapfile     Image location of top down maze image that is in turn converted
                        to gridmap file via the Gridmap class
            num_particles   Number of particles used for tracking
            particle_size   Size for scatter plot particles
            arrow_size      Arrow size (in meters)
        """
        # Set the run
        self.env = env
        # Setup the particles
        self.particles = Particles(num_particles, r_mean, r_std, ang_mean, ang_std)
        
        ## Matplotlib based visualization
        self.visualize = visualize
        self.viz_map   = env.mapfile 

        self.featurizer = HistogramFeatures()

        # Color gradient
        self.num_cgs = 100
        self.color_gradient = list(Color("red").range_to(Color("green"), self.num_cgs))
    
    
    def heatmap(self, viz):
        viz = viz * 0
        planner = self.env.planner
        particles = Particles(planner.poses.shape[0])
        particles.pose[:,:2] = planner.poses[:,:2]
        particles.pose[:, 2] = self.env.agent.pose[:,2]
        pixels = pose2pixel(particles.pose, MC.mazeshape)
        
        weights = np.zeros(len(particles))
        measurement = self.env.get_visual_obs(self.env.agent.pose)
        for i, (pose, pix) in enumerate(zip(particles.pose, pixels)):
            pose_measurement = self.env.get_visual_obs(pose)
            weights[i] = (pose_measurement == measurement).sum()
        #print(weights.max(), weights.min(), weights.mean())
        #weights = softmax(weights, t=.05)
        
        for i, (pose, pix) in enumerate(zip(particles.pose, pixels)):
            c_indx = int((self.num_cgs-1) * weights[i]/weights.max())
            color  = color2bgr(self.color_gradient[c_indx])
            self.env.draw_circle(viz, pix, rad=20, color=color)
            self.env.draw_orientation(viz, pose, thickness=2)
        
        for i, (pose, pix) in enumerate(zip(particles.pose, pixels)):
            cv2.putText(viz, str(int(weights[i])), point(pix), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return viz
    
    def display(self):
        viz = self.env.display().copy()
        pixels = pose2pixel(self.particles.pose, MC.mazeshape)
        for i, (pose, pixel) in enumerate(zip(self.particles.pose, pixels)):
            #viz = self.env.draw_rays(viz, pose)
            #viz = self.env.draw_orientation(viz, pose, thickness=2)

            norm = self.particles.weights[i]/self.particles.weights.max()
            c_indx = int((self.num_cgs-1) * norm)
            color  = color2bgr(self.color_gradient[c_indx])
            viz = self.env.draw_circle(viz, pixel, rad=5, color=color)

        # Make a heatmap
        heatmap = self.heatmap(viz)
        out = np.concatenate((viz, heatmap), 1)
        show(out, 30)
        return out


    def compute_likelihood(self, r, ps, epsilon=1e-12):
        """
        Histogram comparison
        """
        weights = np.zeros((len(self.particles)))
        r = self.featurizer.particle_features(r)
        for i, p in enumerate(ps):
            p = self.featurizer.particle_features(p) 
            weights[i] = self.featurizer.compare(r, p) 
        return weights / weights.sum()
    
    #def compute_likelihood(self, r, ps, epsilon=1e-12):
    #    weights = np.zeros((len(self.particles)))
    #    for i, p in enumerate(ps):
    #        weights[i] = (r * p).sum() 
    #    return weights / weights.sum()
    
    def localize(self):
        
        action = None
        while action != Action.END:
            # Display
            self.display()
            # Get robot measurements
            r_obs = self.env.get_visual_obs(self.env.agent.pose)
            #self.particles.pose[:,2] = self.env.agent.pose[:,2]
            # Get particle measurements
            p_obs = [self.env.get_visual_obs(p) for p in self.particles.pose]
            # Compute similarities between particle measures and real ms.
            self.particles.weights = self.compute_likelihood(r_obs, p_obs)
            # Resample
            self.particles.resample()

            # Move actions
            action = self.env.next_shortest_path_action()
            self.env.makeActions(action)

            # Evolve the particles
            self.particles.random_motion()

        import ipdb; ipdb.set_trace()
    
    #def localize(self):
    #    import ipdb; ipdb.set_trace()
    #    # End frame
    #    t = time.time()
    #    # Loop through robot measurements
    #    len_t = len(self.env)
    #    for ndx in tqdm(range(len_t)):
    #        data = self.env[ndx]
    #        # Get the robot measurement
    #        rm  = data['featurized']
    #        # Get the particle measurements
    #        pms = self.env.pmc[self.particles.pose]['featurized']

    #        # Visualize
    #        if self.visualize:
    #            self.visualize_pf(ndx, data['labelled'], data['topdown'])

    #        # Compute measurement similarities
    #        self.particles.weights = self.compute_similarities(rm, pms) 
    #        # Resample
    #        self.particles.resample()
    #        # Evolve the particles
    #        self.particles.evolve_state()


# Particle visualization model
@click.command()
@click.option('--trial_no', default=None, type=int)
@click.option('--num_particles', default=1000, type=int)
@click.option('--particle_angle_range', default=np.radians(78), type=float)
@click.option('--num_rays', default=60, type=int)
# Particle motion model (in polar coordinates)
@click.option('--r_mean'  , default=0.000, type=float)
@click.option('--r_std'   , default=0.10, type=float)
@click.option('--ang_mean', default=0.000, type=float)
@click.option('--ang_std' , default=np.radians(45), type=float)
# Visualization specifics
@click.option('--visualize', default=True, type=bool)
@click.option('--background', default=True, type=bool)
@click.option('--save_to', default='images', type=str)
# Measurement model
@click.option('--row_min', default=25, type=int)
@click.option('--row_max', default=30, type=int)
@click.option('--n_angle_bins', default=6, type=int)
@click.option('--front_angle_range', default=np.radians(78), type=float)
def run_visual_localization(trial_no, num_particles, particle_angle_range, num_rays, 
                            r_mean, r_std, ang_mean, ang_std, visualize, background, 
                            save_to, row_min, row_max, n_angle_bins, front_angle_range):
    
    data = load_datasets(["train", "val_seen", "test"])
    if trial_no is None:
        trial_no = random.sample(range(len(data)), 1)[0]
    trial = data[trial_no]
    env = RobotSlangSimulator(trial['scan'], show_grid=False)

    # Initialize the visual particle filter
    vpf = ParticleFilter(env, num_particles, particle_angle_range, num_rays, r_mean,
              r_std, ang_mean, ang_std, visualize, background, row_min, save_to,
              row_max, n_angle_bins, front_angle_range)

    # Localize the agent
    vpf.localize()

if __name__ == "__main__":
    run_visual_localization()
