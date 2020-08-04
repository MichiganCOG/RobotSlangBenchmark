"""Label orderings and their corresponding colors
"""
from collections import OrderedDict
import numpy as np

# Label ordering (labels -> classes)
LABELS=dict(red=0,blue=1,green=2,brown=3,yellow=4,
            white=5,floor=6,
            
            jar=7, mug=8, paddle=9)

# Reverse the ordering for CLASSES
CLASSES = {v:f for f,v in LABELS.items()}

# Corresponding label colors
COLORS=dict(red     = np.asarray([  0,   0, 255], dtype=np.uint8),
            blue    = np.asarray([255,   0,   0], dtype=np.uint8),
            green   = np.asarray([  0, 255,   0], dtype=np.uint8),
            brown   = np.asarray([  0,  51, 102], dtype=np.uint8),
            yellow  = np.asarray([  0, 255, 252], dtype=np.uint8),
            white   = np.asarray([255, 255, 255], dtype=np.uint8),
            floor   = np.asarray([  0,   0,   0], dtype=np.uint8),

            jar     = np.asarray([255, 255,   0], dtype=np.uint8),
            mug     = np.asarray([255,   0, 255], dtype=np.uint8),
            paddle  = np.asarray([  0, 128, 128], dtype=np.uint8),)

PIXELS = {tuple(v.tolist()):f for f,v in COLORS.items() }


def color2label(img):
    labelled = np.zeros(img.shape[:-1])

    for pixel,val in PIXELS.items():
        if val in ['ceiling', 'hinge', 'unsure']:
            continue
        mask = np.all(img == np.asarray(pixel), 2)
        labelled[mask] = LABELS[PIXELS[pixel]]
    return labelled

def label2color(img):
    color_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    for c in CLASSES:
        mask = img == c
        color_img[mask] = np.uint8(COLORS[CLASSES[c]])
    return color_img

if __name__ == "__main__":
    import glob
    from os.path import join, abspath, isfile
    import cv2
    maps = sorted(glob.glob('/z/home/shurjo/robotslang/*'))
    maps    = [join(m, 'maze/maze.map.png') for m in maps]
    
    for m in maps:
        if not isfile(m):
            print("{} doesn't exist".format(m))
            continue
        img  = cv2.imread(m)
        labelled = color2label(img)

        m = m.replace('.map.png', '.labelled.npz')
        np.savez_compressed(m, maze_map=labelled)
        print(abspath(m))









