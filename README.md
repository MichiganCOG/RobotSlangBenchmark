## RobotSlang Benchmark README

Use the following script to setup a virtual environment and cache shortest paths and parrticle measurements.

```
source install.sh 
```

## Training
Use the provided `train.py` script to train the baseline seq2seq model. The script takes the following options:

| Option  | Possible values  |
|---|---|
| `feedback`  |  'sample', 'teacher' |
| `eval_type`  | 'val', 'test'  |
| `blind`  | 'vision', 'language', ''  |

Note that in the traditional experiment where no blinding occurs, the `blind` parameter is blank. As an example, to train a model with teacher forcing, use the following command. 
```
python train.py --feedback=teacher --eval_type=val --lr=0.0001
```
Note that your device must be CUDA-enabled. 


## Visualizations
To make videos of our trials using the simulator run the following:
```
python make_simulation_videos.py 
```
