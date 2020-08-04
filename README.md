#### RobotSlang Benchmark README

Use the following script to setup a virtual environment and cache shortest paths and parrticle measurements.

```
source setup.sh
```

## Train and Evaluate
Use the provided `train.py` script to train the baseline seq2seq model. The script takes the following options:

| Option  | Possible values  |
|---|---|
| `feedback`  |  'sample', 'teacher' |
| `eval_type`  | 'val', 'test'  |


