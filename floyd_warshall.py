"""
"""
import sys; import os; sys.path.append(os.getcwd())
import os
import cv2
import glob
import random
from constants import MazeConstants as MC, MotionModel as MM
from rslang_utils import show, pixel2pose, pose2pixel, point, angle_between, round_angle
from labels import label2color
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import cdist
from os.path import join, dirname, isfile

class FloydWarshallRobotslang:

    def __init__(self, mapfile, radius=MM.R):
       
        self.root    = dirname(mapfile)
        self.mapfile = cv2.imread(mapfile)
        self.radius  = radius
        # Load cached data
        self.cache_file = join(self.root, 'sp_FW.npz')
        if not isfile(self.cache_file):
            self.cache_sps(self.mapfile, radius)
        # Init variables
        self.cache_var = self.poses_var = self.pixels_var = self.dist_matrix_var = None
        self.predecessors_var = None 
   
    @property
    def cache(self):
        if self.cache_var is None:
            with np.load(self.cache_file) as d:
                self.cache_var  = d['cache']
        return self.cache_var

    @property
    def poses(self):
        if self.poses_var is None:
            with np.load(self.cache_file) as d:
                self.poses_var = d['poses']
        return self.poses_var
    
    @property
    def pixels(self):
        if self.pixels_var is None:
            with np.load(self.cache_file) as d:
                self.pixels_var = d['pixels']
        return self.pixels_var
    
    @property
    def dist_matrix(self):
        if self.dist_matrix_var is None:
            with np.load(self.cache_file) as d:
                self.dist_matrix_var = d['dist_matrix']
        return self.dist_matrix_var
    
    @property
    def predecessors(self):
        if self.predecessors_var is None:
            with np.load(self.cache_file) as d:
                self.predecessors_var = d['predecessors']
        return self.predecessors_var
    
    @property
    def navigable_locations(self):
        return self.poses
    
    @property
    def navigable_pixels(self):
        return self.pixels

    def grid(self, radius):
        # Make a grid
        x = np.arange(0, MC.width , radius)
        y = np.arange(0, MC.height, radius)

        xx, yy = np.meshgrid(x,y)
        xx, yy = xx.flatten(), yy.flatten()

        poses = np.stack([xx, yy], 1)
        pixels = pose2pixel(poses, MC.mazeshape)

        return poses, pixels

    def remove_colored_pixels(self, mapfile, poses, pixels):
        # Remove pixels that exist on a color
        where = np.where(mapfile[pixels[:,0], pixels[:,1]].sum(1) == 0)
        poses  = poses[where]
        pixels = pixels[where]
        return poses, pixels
    
    def remove_close_to_walls(self, mapfile, poses, pixels, radius): 
        # Remove pixels that are within a certain radius of the wall
        # Utilizes the fact that blacke pixels correspond to no collision
        pix_radius = int(round(MC.mazeshape[1]/MC.width * radius))
        where = []
        for i,p in enumerate(pixels):
            r,c = p
            r_min = max(r - pix_radius, 0)
            r_max = min(r + pix_radius, MC.mazeshape[0])
            c_min = max(c - pix_radius, 0)
            c_max = min(c + pix_radius, MC.mazeshape[1])
            square = mapfile[r_min:r_max, c_min:c_max]
            if square.sum() == 0: # check if pixel is black
                where.append(i)
        where = np.asarray(where)
        poses = poses[where]
        pixels = pixels[where]

        return poses, pixels

    def get_distance_matrix(self, poses, radius):
        # Makes adjacency matrix and runs floyd warshall
        distances = cdist(poses, poses)
        mask = distances > np.sqrt(2) * radius
        distances[mask] = 0
        graph = distances
        dist_matrix, predecessors = floyd_warshall(csgraph=graph, 
                                                   directed=False, 
                                                   return_predecessors=True)
        return dist_matrix, predecessors

    
    def get_shortest_path(self, node_a, node_b):
        shortest_path = []
        def recurse_shortest_path(i, j):
            if i != j:
                recurse_shortest_path(i, self.predecessors[i,j])
            shortest_path.append(j)
        recurse_shortest_path(node_a, node_b)
        return shortest_path
    
    
    def closest_node(self, pose):
        pose   = np.expand_dims(np.squeeze(pose)[:2], 0)
        dists  =  np.linalg.norm(self.poses - pose, axis=1)
        return dists.min(), dists.argmin()

    def closest_node_pose(self, pose):
        # Closest navigable node by eucldian disntance
        am = self.closest_node(pose)[1]
        return self.poses[am]

    
    def get_euc_dist(self, start, end):
        return np.linalg.norm(np.squeeze(start)[:2] - np.squeeze(end)[:2])

    def __call__(self, start, end):
        _, start = self.closest_node(start)
        _, end   = self.closest_node(end)
        return self.get_traj_from_ids(start, end)
    
    def next_node(self, start, end):
        _, start = self.closest_node(start)
        _, end   = self.closest_node(end)
        return self.cache[start, end]

    def get_traj_from_ids(self, start, end):
        traj  = self.get_shortest_path(start, end)
        if len(traj) > 1:
            return self.poses[traj]
        else:
            return None
    
    def traj_dist_meters(self, traj):
        if len(traj) <= 1:
            return 0
        else:
            return np.linalg.norm(traj[:-1] - traj[1:], axis=1).sum()
    
    def get_top_dist(self, start, end):
        _, start = self.closest_node(start)
        _, end   = self.closest_node(end)
        return self.dist_matrix[start, end]
    
    def cache_sps(self, mapfile, radius):
        poses, pixels = self.grid(radius)
        poses, pixels = self.remove_colored_pixels(mapfile, poses, pixels)
        poses, pixels = self.remove_close_to_walls(mapfile, poses, pixels, radius)
    
        self.dist_matrix_var, self.predecessors_var = \
                self.get_distance_matrix(poses, radius)
        
        self.poses_var  = poses
        self.pixels_var = pixels
        # Precompute next node for all shortest paths
        # Next node in turn is used to compute next action
        num_nodes = len(poses)
        cache = np.zeros((num_nodes, num_nodes, 2))
        for i in range(num_nodes):
            for j in range(i):
                if not np.isinf(self.dist_matrix[i,j]) and i != j:
                    traj = self.get_shortest_path(i, j)
                    cache[i, j] = poses[traj[1]]
                    cache[j, i] = poses[traj[-2]]

        # Save the cache files for easy use later
        np.savez_compressed(self.cache_file, 
                            cache=cache,
                            poses=poses,
                            pixels=pixels,
                            dist_matrix=self.dist_matrix, 
                            predecessors=self.predecessors)

if __name__ == "__main__":
    # Shortest paths between all nodes uses floyd warshall
    from constants import Globs
    import glob
    from rslang_utils import multiprocess_progress_bar
    import multiprocessing
    from multiprocessing import Pool
    import tqdm
    
    print("Caching shortest paths")
    def cache_shortest_path(mapfile):
        planner = FloydWarshallRobotslang(mapfile)
    maps = glob.glob(Globs.raw_maps)
    num_proc = max(multiprocessing.cpu_count()//2, 1)
    with Pool(num_proc) as p:
        list(tqdm.tqdm(p.imap(cache_shortest_path, maps), total=len(maps)))
    print()
