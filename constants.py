import numpy as np
import sys
from os.path import join
import os

class MazeConstants:
    # Geometery
    width  = 2.4638 #x_range
    height = 2.0574 #y_range
    mazeshape = (1175, 1401, 3) 

    # Particle size
    PRAD = 10
    ORAD = 40

    RDIMS = PRAD

class DriverConstants:
    angle_of_view = np.radians(78)
    #angle_of_view = np.radians(78)
    img_shape     = ( 720, 1280, 3)


class CommanderConstants:
    img_shape = (1175, 1401, 3) 

class MotionModel:
    R = 0.07
    T = np.radians(45)

class EpisodeConstants:
    max_len = 120

class  Action:
    FORWARD  = 0
    RIGHT    = 1
    LEFT     = 2
    END      = 3

class Files:
    mapfile   = 'maze.map.png'
    objfile   = 'maze.map-fixed.png'
    basedir   = os.getcwd()
    data      = basedir + '/' + 'NDH-data'
    results   = basedir + '/' + 'NDH-output/results'
    plots     = basedir + '/' + 'NDH-output/plots'
    snapshots = basedir + '/' + 'NDH-output/snapshots'

class Globs:
    raw_maps = Files.basedir + '/' + 'mazedata/Run_*/maze/{}'.format(Files.mapfile)
    obj_maps = Files.basedir + '/' + 'mazedata/Run_*/maze/{}'.format(Files.objfile)
