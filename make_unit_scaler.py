import json
from rslang_simulator import RobotSlangSimulator, resize50
import time
import cv2
from multiprocessing import Pool
import numpy as np
from sklearn.preprocessing import StandardScaler
from constants import Files

def get_data(scan):
    env = RobotSlangSimulator(scan)
    arr = env.get_all_data() 
    return arr 

class X:
    def map(self, *args):
        return list(map(*args))

class FakeScaler:
    def transform(self, x):
        return x

def get_scaler(debug):
    if False: #debug:
        print("BAD SCALER")
        return FakeScaler() 
    else:

        # Get the measurements 
        with open('{}/train.json'.format(Files.data)) as f:
            data = json.load(f)

        # Get the measurements from different mazes
        scans = [data[i]['scan'] for i in range(len(data))]
        num_processes = 6
        #p = X()
        p = Pool(num_processes)
        measurements = p.map(get_data, scans)
        measurements = [m for m in measurements if len(m) > 1 ]

        # Concatenate and run unit scale
        measurements = np.concatenate(measurements, 0) 
        scaler = StandardScaler()
        scaler.fit(measurements)

        return scaler


if __name__ == "__main__":
    scaler = get_scaler()
    import ipdb; ipdb.set_trace()

    

