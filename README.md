# Sub-trajectory Clustering with Deep Reinforcement Learning
This is a python implementation of the paper 'Sub-trajectory Clustering with Deep Reinforcement Learning'.

## Requirements
- Ubuntu OS
- python==3.6 (Anaconda3 is recommended)
- tensorflow==2.2.0
- keras==2.4.3
- scikit-learn==0.24.2
- tqdm==4.62.0
- numpy==1.19.2
- Datasets can be downloaded from [here](https://drive.google.com/file/d/1LNIPnAAfyZNBaUvRoV6ghITCcgaMJ-sw/view?usp=drive_link), and tar -xzvf data.tar.gz

## Preprocessing
```
cd subtrajcluster/
python preprocessing.py
```

You could directly use our provided datasets.

## Training
```
python rl_train.py
```

You could use n-fold validations to train models, the default value of n is 5. 
```
python crosstrain.py
```

The trained models can be downloaded from [here](https://drive.google.com/file/d/19l8UqEIT2Z5ndTKLq3_Z4xClKbHtAZMq/view?usp=drive_link), and tar -xzvf savemodels.tar.gz

## Experiments
Please refer to the [experiments](./experiments/readme.md) for the experiments.
