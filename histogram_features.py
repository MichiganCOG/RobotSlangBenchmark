import cv2
import numpy as np
import time 
from constants import DriverConstants as DC
from labels import LABELS

class HistogramFeatures:
    def __init__(self, n_angle_bins=6, ang_min=-DC.angle_of_view/2, ang_max=DC.angle_of_view/2):
        
        # Angle ranges equal to that of camera lens
        self.ang_min, self.ang_max = ang_min, ang_max

        # Angles/column bins
        a_bins = np.linspace(ang_min, ang_max, n_angle_bins+1)
        # Class bins
        class_bins = np.arange(0, len(LABELS) + 1) - .5
        # Actual bin boundaries
        self.ctheta_bins = [class_bins, a_bins]
        # Add an angle 
        self.front_angles = None 
        # Particle observations
        self.num_particles = None
    
    def augment_driver(self, img):
        # Image obseration (for hist featurization)
        if self.front_angles is None:
            front_angles = np.linspace(self.ang_max, self.ang_min, DC.img_shape[1])
            self.front_angles = np.tile(front_angles, (img.shape[0], 1))
        img = np.stack((img, self.front_angles), len(img.shape) + 1)
        return img.reshape((-1, 2))

    def driver_features(self, img):
        img = self.augment_driver(img) 
        hist = np.histogramdd(img, bins = self.ctheta_bins)[0]
        hist /= np.maximum(hist.sum(0, keepdims=True), 1e-12)
        return hist

    def augment_particles(self, img):
        if not self.num_particles:
            self.num_particles = img.shape[0]
            ndxs = np.expand_dims(np.arange(img.shape[0]), 1)
            ndxs = np.repeat(ndxs, img.shape[1], axis=1)
            self.ndxs = ndxs
            self.ndx_bins = np.arange(img.shape[0]+1) -.5 

        front_angles = np.linspace(self.ang_max, self.ang_min, img.shape[1]).reshape(1, -1)
        img = np.stack((self.ndxs, img, front_angles), axis=2)

        return img.reshape((-1, 3))

    def particle_features(self, img):
        """
        Histogram method applied in batch"""
        if len(img.shape) == 1:
            img = img.reshape(1, -1)
        imgs = self.augment_particles(img) 
        hist = np.histogramdd(imgs, bins=[self.ndx_bins, *self.ctheta_bins])[0]
        hist /= np.maximum(hist.sum(1, keepdims=True), 1e-12)
        return hist.reshape(hist.shape[0], -1)

    def compare(self, h1, h2):
        """
        HISTCMP_CORREL        : Correlation
        HISTCMP_CHISQR        : Chi-Square
        HISTCMP_CHISQR_ALT    : Alternative Chi-Square
        HISTCMP_INTERSECT     : Intersection
        HISTCMP_BHATTACHARYYA : Bhattacharyya distance
        HISTCMP_HELLINGER     : Synonym for CV_COMP_BHATTACHARYYA
        HISTCMP_KL_DIV        : Kullback-Leibler divergence
        """
        h1 = np.float32(h1.flatten())
        h2 = np.float32(h2.flatten())
        return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)


