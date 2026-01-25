# DeepLearning_VizDoom


```
conda env create -f environment.yml
conda activate vizdoom-dm

# start training (indefinitely, stop with ctrl+c)
python -m project.main --train

# save video (can be run while training)
python -m project.main --save

# start tensorflow dashboard (can be run while training)
tensorboard --logdir=./train_dir --port=6006
```


