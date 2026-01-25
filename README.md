# DeepLearning_VizDoom


```
conda env create -f environment.yml
conda activate vizdoom-dm

# start training (indefinitely, stop with ctrl+c)
python --train

# save video (can be run while training)
python --save

# start tensorflow dashboard (can be run while training)
tensorboard --logdir=./train_dir --port=6006
```


