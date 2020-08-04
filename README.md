## RobotSlang Benchmark README

Use the following script to setup a virtual environment and cache shortest paths and parrticle measurements.

```
source setup.sh
```

## Training
Use the provided `train.py` script to train the baseline seq2seq model. The script takes the following options:

| Option  | Possible values  |
|---|---|
| `feedback`  |  'sample', 'teacher' |
| `eval_type`  | 'val', 'test'  |
| `blind`  | 'vision', 'language', ''  |

For example, to train a model with teacher forcing, 
```
python train.py --feedback=teacher --eval_type=val --blind= --lr=0.0001
```
