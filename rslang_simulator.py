"""
Make a simple simulator that uses particles
"""
import sys; import os; sys.path.append(os.getcwd())
import os
from particles import Particles
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np
from labels import label2color
from rslang_utils import pose2pixel, point, pixel2pose, to_onehot, to_onehot_array
import click
from os.path import join, basename, dirname
from lut import LUT
from rslang_utils import numify, resize50, to_array, show, angle_between, round_angle, l2
import json
from constants import MazeConstants as MC, MotionModel as MM, EpisodeConstants as EC, Files
from functools import lru_cache
from labels import COLORS, color2label, LABELS
import random
from skimage.draw import line
from constants import Action
from floyd_warshall import FloydWarshallRobotslang

# Colors in BGR format
WHITE = (255, 255, 255)
RED   = (  0,   0, 255)
BLUE  = (255,   0,   0)
GREEN = (  0, 255,   0)

@lru_cache()
def get_targets(fname=Files.basedir + '/lookup_scans.json'):
    with open(fname, 'r') as f:
        out = json.load(f)
    return out

class RobotSlangSimulator:

    def __init__(self, scanId, show_rays=True, show_grid=True): 
        self.scanId = scanId
        self.root, self.target, agent_pose, goal_pose = scanId.split('-')
        self.root = os.path.join(Files.basedir, dirname(self.root))
        self.agent_pose_ini = to_array(agent_pose)
        self.goal_pose_ini  = to_array(goal_pose)
        
        # Cache variables
        self.pmc_var = None
        # Visualization settnigs
        self.show_rays = show_rays
        self.show_grid = show_grid 
        
        # Floyd warshall algorithm for shortest path supervision
        self.mapfile_var = join(self.root, Files.objfile)
        self.planner = FloydWarshallRobotslang(self.mapfile_var)
        
        # Navigation pixels for  visualization
        self.nav_pixels   = self.planner.navigable_pixels

        ## Target poses
        self.target_poses  = get_targets()[scanId]
        self.target_pixels = pose2pixel(np.asarray(self.target_poses), MC.mazeshape)
        self.target_pixels = list(zip(["jar", "mug", "paddle"], self.target_pixels))

        # Randomize location of agent and goal
        self.agent   = Particles(1)
        self.goal    = Particles(1)

        # Start by resetting
        self.reset()

    @property
    def pmc(self):
        if self.pmc_var is None:
            obj_map = join(self.root, Files.objfile) 
            self.pmc_var = LUT(obj_map)
        return self.pmc_var

    @property
    def mapfile(self):
        if type(self.mapfile_var) == str:
            self.mapfile_var = cv2.imread(self.mapfile_var)
        return self.mapfile_var

    def face_object(self):
        """Start trials facing the target object"""
        self.step_counter = -1
        
        # Move away from starting pose / object
        for i in range(20):
            action = self.next_shortest_path_action()
            self.makeActions(action)
            self.agent.pose = self.cnl(self.agent.pose)
        
        # Find starting object pose
        ndx = 0 if self.target == 'mug' else 1
        back_to = self.target_poses[ndx]

        # Face starting object 
        while l2(back_to, self.agent.pose) >= 3*MM.R:
            action = self.next_shortest_path_action(self.agent.pose, back_to) 
            self.makeActions(action)
            self.agent.pose = self.cnl(self.agent.pose)
        
    def reset(self):
        self.goal.pose[ ...,:2] = self.goal_pose_ini
        self.goal.pose[:]  = self.cnl(self.goal.pose)
        
        # Set poses
        if self.target == 'jar':
            self.agent.pose[:] = self.agent_pose_ini[:]
            self.agent.pose[:] = self.cnl(self.agent.pose)
        else:
            self.agent.pose[...,:2] = self.agent_pose_ini
            self.agent.pose[:] = self.cnl(self.agent.pose)
            self.face_object()
        
        # Reinit the step counte
        self.step_counter = -1
    
    def cnl(self, pose):
        """Closes Navigable Location according to choices of discretization"""
        pose[:,:2] = self.planner.closest_node_pose(pose)
        if pose.shape[-1] == 3:
            pose[:,2] = round_angle(pose[:,2], MM.T)
        return pose
    
    def check_collision(self):
        R = self.getR()
        new_pose = self.agent.next_pose(R, 0)
        collision = self.planner.get_top_dist(self.agent.pose, new_pose)
        collision  = collision > .11 # if not a side or hypotenuse away 
        return collision

    def draw_circle(self, viz, px, rad=MC.ORAD, color=BLUE):
        cv2.circle(viz, point(px), rad, tuple([int(c) for c in color]), thickness=-1)
        return viz

    def display(self, viz=None, draw_measurements=False):
        # Draw agent
        viz = self.mapfile.copy() if viz is None else viz.copy()
        
        # Draw the navigable locations
        if self.show_grid:
            for p in self.nav_pixels:
                self.draw_circle(viz, p,  rad=5, color=(128, 128, 0))
        
        # Draw rays cast from the agent
        if self.show_rays:
            self.draw_rays(viz, self.agent.pose)
        
        # Draw the orientation
        self.draw_orientation(viz, self.agent.pose)
        
        # Draw the agent
        self.draw_agent(viz, self.agent.pose)
        
        # Draw the trajectory
        traj = self.planner(self.agent.pose, self.goal.pose)
        self.draw_trajectory(viz, traj)
        
        if draw_measurements:
            # Add visualization lines
            colors, depths = self.draw_measurement_lines()
            # Resize for visualization
            itype = cv2.INTER_NEAREST
            colors = cv2.resize(colors, (viz.shape[1], 100), interpolation=itype)
            depths = cv2.resize(depths, (viz.shape[1], 100), interpolation=itype)
            viz = np.concatenate((viz, colors, colors * 0, depths), 0)

        return viz

    def draw_agent(self, viz, pose):
        stp, enp = self.orientation_vector(pose)
        st_pix = pose2pixel(stp, viz.shape)
        self.draw_circle(viz, st_pix, rad=MC.PRAD*2, color=BLUE)

    def orientation_vector(self, pose, r=.03):
        stp = np.squeeze(pose)
        enp = stp[:2] + r * np.asarray([np.cos(stp[2]), np.sin(stp[2])])
        return stp, enp
    
    def draw_orientation(self, viz, pose, thickness=10): 
        # Start and end pose for arrow
        stp, enp = self.orientation_vector(pose)

        # Convert to pixel
        st_pix = pose2pixel(stp, viz.shape)
        en_pix = pose2pixel(enp, viz.shape)
        
        # Draw orientation
        cv2.arrowedLine(viz, point(st_pix), point(en_pix), GREEN, thickness=thickness)
        return viz
    
    def draw_rays(self, viz, pose):
        stp, enp = self.orientation_vector(pose)
        st_pix = pose2pixel(stp, viz.shape)
        out = self.pmc.get_rays(pose)
        for o in out:
            cv2.line(viz, point(st_pix), point(o), WHITE, thickness=4)
        return viz

    def draw_measurement_lines(self):
        # Add measurement below
        color = self.get_visual_obs(self.agent.pose)
        measurement = np.expand_dims(color, 0)
        measurement = label2color(measurement)

        depth = self.get_depth_obs(self.agent.pose)
        depth /= max(depth.max(), 1e-12)
        depth = depth.reshape(1,-1, 1)
        cvec = 255*np.ones((1, color.shape[0], 3))
        cvec = np.uint8(255 - depth * cvec)

        return measurement, cvec

    def get_input(self):
        c = click.getchar()
        if   c == '\x1b[A': #forward
            action = Action.FORWARD
        elif c == '\x1b[C': #left
            action = Action.LEFT
        elif c == '\x1b[D': #right
            action = Action.RIGHT
        else:
            print("Not supported")
            action = 3
        #elif c == '\x1b[B': #backward
        #    action = Action.BACKWARD

        return action
    
    def play(self):
        while True:
            show(self.display(draw_measurements=True), 30)
            action = self.get_input()
            self.makeActions(action)

    
    def getR(self):
        # Hypotenuse or straight motion
        degrees = np.abs(np.degrees(self.agent.pose[:,2]) - np.asarray([0, 90, 180, 270, 360]))
        
        if degrees.min() < 1 : 
            R = MM.R
        else:
            R = MM.R * np.sqrt(2)
        return R
    
    #def next_shortest_path_action(self):
    #    """RANDOM BASELINE"""
    #    if self.step_counter < 119:
    #        if not self.check_collision():
    #            action = random.sample([Action.FORWARD, Action.LEFT, Action.RIGHT], 1)[0]
    #        else:
    #            action = random.sample([Action.LEFT, Action.RIGHT], 1)[0]
    #        return action
    #    else:
    #        return Action.END 
    
    def next_shortest_path_action(self, pose1=None, pose2=None):
        """teacher action"""
        if pose1 is None:
            pose1 = self.agent.pose
        if pose2 is None:
            pose2 = self.goal.pose
        next_node = self.planner.next_node(pose1, pose2)
        if next_node.sum() > 0:
            action = self.get_action(next_node)
            return action 
        else:
            return Action.END 
    
    def get_action(self, next_node):
        
        if self.step_counter < EC.max_len:
            stp, enp = self.orientation_vector(self.agent.pose)
            angle    = angle_between(next_node - stp[:2], enp - stp[:2])
            
            if np.abs(angle) < np.radians(1):
                action = Action.FORWARD
            elif np.sign(angle) == 1:
                action = Action.LEFT
            else:
                action = Action.RIGHT
        else:
            # End the episode after step counter threshold passed
            action = Action.END

        return action
    
    def makeActions(self, action):
        self.step_counter += 1
        R = self.getR()

        if action == Action.FORWARD:   # Forward
            dr = R
            da =  0
        elif action == Action.RIGHT: # Right
            dr = 0
            da = MM.T
        elif action == Action.LEFT: # Left
            dr = 0
            da = -MM.T 
        else:
            dr = da = 0
        
        # move the agent
        self.agent.move(dr, da)
        self.agent.pose[:] = self.cnl(self.agent.pose)
        
        #self.visualize_every_episode()
    
    def get_visual_obs(self, pose):
        colors = self.pmc[pose]
        return colors
    
    def get_depth_obs(self, pose):
        rays   = self.pmc.get_rays(pose)
        agent = self.agent.pose[:,:2]
        poses = pixel2pose(rays, MC.mazeshape)
        dists = np.linalg.norm(agent-poses, axis=1)
        return dists 
    
    def get_obs(self, pose=None):
        if pose is None:
            pose = self.agent.pose
        colors = self.get_visual_obs(pose).flatten() 
        colors = to_onehot_array(colors, len(LABELS)).flatten()
        dists = self.get_depth_obs(pose).flatten()
        out = np.concatenate((colors, dists, [self.check_collision()]))
        return out

    def draw_trajectory(self, viz, traj_poses, color=(59,181,207)):
        if traj_poses is not None:
            traj_pix = pose2pixel(traj_poses, viz.shape)
            for i in range(len(traj_pix)-1):
                cv2.line(viz, point(traj_pix[i]), point(traj_pix[i+1]), color, thickness=4)
            return viz
     
    
    def __repr__(self):
        return "{}-{}".format(numify(self.root), self.target)
    
    def shortest_agent(self):
        action = None
        while action != Action.END:
            action = self.next_shortest_path_action()
            self.makeActions(action)
        return self.step_counter

    def shortest_agent_images(self):
        action = None
        while action != Action.END:
            yield self.display(draw_measurements=True) 
            action = self.next_shortest_path_action()
            self.makeActions(action)
    
    def get_all_data(self):
        """
        Used to make a zero mean / unit variance scaler
        """
        data = []
        action = None
        while action != Action.END:
            data.append(self.get_obs(self.agent.pose))
            # Move along shortest path
            action = self.next_shortest_path_action()
            self.makeActions(action)
        return np.asarray(data)
    
    def make_video(self, folder):
        width  = 700
        height = 588
        fps    = 30
        fourcc = VideoWriter_fourcc(*'MP42')
        vfile  = '{}/{}.avi'.format(folder, str(self)) 
        video  = VideoWriter(vfile, fourcc, float(fps), (width, height))
        
        traj = self.planner(self.agent.pose, self.goal.pose)
        
        while True:
            img = self.display()
            self.draw_trajectory(img, traj)
            img = resize50(img)
            video.write(img)
            # Move along shortest path
            action = self.next_shortest_path_action()
            self.makeActions(action)
            if action == Action.END:
                break

        video.release()

        print('Video saved to: {}'.format(vfile))
        return self.step_counter


if __name__ == "__main__":
    from rslang_utils import load_data
    random_trial = random.sample(load_data(), 1)[0]
    trial = RobotSlangSimulator(random_trial['scan'])
    trial.play()
    import ipdb; ipdb.set_trace()
    
