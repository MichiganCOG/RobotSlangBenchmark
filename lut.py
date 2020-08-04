"""
Look up table for particles index by (x, y, theta).
The image is treated as an occupancy grid where each
pixel is assumed to be cell.
"""
import sys; import os; sys.path.append(os.getcwd()) #FIXME weird bug makes this necessary
import numpy as np
import cv2
import os
from os.path import join, isfile, abspath, dirname
from skimage.draw import line
from functools import lru_cache
from labels import LABELS, color2label 
from rslang_utils import pose2pixel, point, angle_range_vector
from constants import MazeConstants as MC, DriverConstants as DC
import numpy as np

def find_walls(p1, angle):
    # 4 corner vectors
    c1 = -p1
    c2 = np.asarray([[MC.width,         0]]) - p1
    c3 = np.asarray([[MC.width, MC.height]]) - p1
    c4 = np.asarray([[       0, MC.height]]) - p1

    # Compute angles between first corner vec with everything else
    # the order of p2's angle determines what wall is hit
    p2 = np.stack((np.cos(angle), np.sin(angle)), 1)
    ref_vecs = np.stack([p2, c2, c3, c4], 1)
    c1 = np.expand_dims(c1, 1)

    # Batch-wise cross product and dot product for angle computation 
    cosangle = (c1 * ref_vecs).sum(2) #dot product
    sinangle = c1[...,0] * ref_vecs[...,1] - c1[...,1] * ref_vecs[...,0] #cross product

    # Angle 
    angles = np.arctan2(sinangle, cosangle)
    angles[angles < 0] += 2*np.pi
    
    # Sort the angles (wherever the 1st index moves is the wall)
    sindxs = np.argsort(angles, 1)
    
    # Find ordering the p2 (index 0) angle with respect to other corners
    return np.argmin(sindxs, 1)

def solve_x(xy, Y, m, b, eps=1e-16):
    xy[:,1] = Y
    xy[:,0] = (Y-b) / (m + eps)
    return xy

def solve_y(xy, X, m, b):
    xy[:,0] = X
    xy[:,1] = m * X + b
    return xy

def interesection_with_walls(p1, angle, eps=1e-16):
    """
    Given rays and heading, finds which maze wall it intersects with
    """
    # Compute the equation of the rays (slope and intercept)
    ms = np.tan(angle)
    bs = p1[:,1] - ms * p1[:,0]
    
    # Extract the coordinates corresponding to wall interesection
    wall_ndxs = find_walls(p1, angle)
    # Find intersection with the walls
    wall_pts = np.zeros_like(p1)
     
    # Lower wall; y = 0;
    for ndx in range(4):
        mask = wall_ndxs == ndx
        if ndx in [0,2]:
            wall_pts[mask] = solve_x(wall_pts[mask], 
                                     Y=0 if ndx==0 else MC.height, 
                                     m=ms[mask], 
                                     b=bs[mask])
        else:
            wall_pts[mask] = solve_y(wall_pts[mask], 
                                     X=0 if ndx==3 else MC.width, 
                                     m=ms[mask], 
                                     b=bs[mask])
    return wall_pts


def sample_uniform(disc_step, disc_angle):
    # Sample points from a uniform grid
    x   = np.arange(0, MC.width , disc_step)
    y   = np.arange(0, MC.height, disc_step) 
    ang = np.arange(0, 2*np.pi, disc_angle)
    # Meshgrid
    xx, yy, aa = np.meshgrid(x, y, ang)
    shape = xx.shape
    xx, yy, aa = list(map(lambda l: l.flatten(), (xx, yy, aa)))
    # Form the starting points
    return np.stack([xx, yy], 1), aa, shape



@lru_cache()
def pixel_rays(mapshape, disc_step, disc_angle):
    # Find intersection of points with the wallsi
    points_start, angles, shape = sample_uniform(disc_step, disc_angle)
    points_end   = interesection_with_walls(points_start, angles)
    # Convert to pixels
    pixels_start = pose2pixel(points_start, mapshape)
    pixels_end   = pose2pixel(points_end, mapshape)
    # Extract coordinates of lines between start and end points i.e. the rays
    pixel_lines = [line(*p1, *p2) for p1,p2 in zip(pixels_start, pixels_end)]
    return pixel_lines, shape


def make_cache(mapfile, disc_step, disc_angle):
    # Convert mapfiles to labels (based on color)
    mapfile = color2label(cv2.imread(mapfile))

    # Pixel points at the end of rays
    pixel_lines, shape = pixel_rays(mapfile.shape, disc_step, disc_angle)
    # Find rays end points and extract 
    pixels_end = np.zeros((len(pixel_lines), 2), dtype=np.int32) 
    for i, (r,c) in enumerate(pixel_lines):
        mask = (mapfile[r,c] == LABELS['floor'])
        p = np.argmin(mask)
        pixels_end[i][0] = pixel_lines[i][0][p]
        pixels_end[i][1] = pixel_lines[i][1][p]
    # Extract the labels at the end of rays
    pixels_vals = mapfile[pixels_end[:,0], pixels_end[:,1]]

    # Reshape for indexing purposes
    pixels_end  = pixels_end.reshape((*shape, 2))
    pixels_vals = pixels_vals.reshape(*shape)

    return pixels_vals, pixels_end 
    

class LUT:
    """Look up table
    Args:
         dics: Pixel discretization
    """
    def __init__(self, mapfile, disc_step=.05, disc_angle=np.radians(6)):
        # Discretization steps
        self.disc_step  = disc_step
        self.disc_angle = disc_angle
        
        # Check if cache exists otherwise make it
        self.cfile = join(dirname(mapfile), 'measurements.npz') 
        
        # Cache the pixel and rays measurements
        self.cache_var = self.rays_var = None
        self.mapshape = (1175, 1401)
        aov = DC.angle_of_view
        self.front_angle_range = np.arange(-aov/2, aov/2, self.disc_angle)
        if not isfile(self.cfile):
            cache, rays = make_cache(mapfile, self.disc_step, self.disc_angle)
            np.savez_compressed(self.cfile, cache=cache, rays=rays)
    
    @property
    def cache(self):
        if self.cache_var is None:
            with np.load(self.cfile) as d:
                self.cache_var = d['cache']
        return self.cache_var

    @property
    def rays(self):
        if self.rays_var is None:
            with np.load(self.cfile) as d:
                self.rays_var = d['rays']
        return self.rays_var

    def pose_to_disc(self, pose):
        i_x = np.int32(pose[...,1]/self.disc_step)
        i_y = np.int32(pose[...,0]/self.disc_step)
        i_t = np.int32(pose[...,2]/self.disc_angle)
        return i_x, i_y, i_t 
    
    def rays_angles(self, pose):
        angles   = angle_range_vector(pose[...,2] + self.front_angle_range)
        return  np.int32(angles/self.disc_angle) # indices

    def __getitem__(self, pose):
        i_x, i_y, i_t = self.pose_to_disc(pose)
        return self.cache[i_x, i_y, self.rays_angles(pose)]

    def get_rays(self, pose):
        i_x, i_y, i_t = self.pose_to_disc(pose)
        return self.rays[i_x, i_y, self.rays_angles(pose)]


if __name__ == "__main__":
    # Cache the particle measurements 
    from constants import Globs
    import glob
    from rslang_utils import multiprocess_progress_bar
    import multiprocessing
    from multiprocessing import Pool
    import tqdm
    
    print("Caching measurements")
    def cache_LUT(mapfile):
        planner = LUT(mapfile)
    maps = glob.glob(Globs.obj_maps)
    num_proc = max(multiprocessing.cpu_count()//2, 1)
    with Pool(num_proc) as p:
        list(tqdm.tqdm(p.imap(cache_LUT, maps), total=len(maps)))
    print()
