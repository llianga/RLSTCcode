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
- Datasets can be downloaded from [here](https://jbox.sjtu.edu.cn/l/t1Vb6f), and tar -zxvf data.tar.gz -C ./data/

## Preprocessing
To get the required data or you can directly use our provided dataset.

```
cd subtrajcluster/
python preprocessing.py
```

You could preprocess other datasets.

## Training
You could use n-fold validations to train models, the default value of n is 5. 

```
python crosstrain.py
```

You can also train the models using a non n-fold validation method.

```
python rl_train.py
```

- Models can be downloaded from [here](https://jbox.sjtu.edu.cn/l/71xmzT), and tar -zxvf models.tar.gz -C ./models/

## Experiments
