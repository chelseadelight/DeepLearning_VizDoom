# DeepLearning_VizDoom

# Quick Start (Local)

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

# Quick Start (Slurm HPC)
```
# create the conda environment
cd slurm
./setup.sh

# run training as a slurm job
# optional flags can be appended
sbatch train.sh \
--train \
--architecture=gru \
--workers=16 \
--worker-envs=16
```

# Custom Model Architectures

To use a custom model architecture, create a new file in the `project/models` directory using the existing model files as templates. Then, specify your custom model when running the training script by using the `--model` argument followed by the name of your custom model file (without the `.py` extension). For example: 

```
python -m project.main --train --model custom_model
```

It is likely that you will also need to provide the `--model` flag when saving videos to ensure the correct model architecture is used for inference:

```
python -m project.main --save --model custom_model
```

# Custom Environments

To add custom ViZDoom environments, create a new `DoomSpec` instance in the `CUSTOM_ENVS` list located in `project/main.py`. You can use the existing environment specifications as templates. Make sure to import any necessary wrappers or configurations at the top of the file.

Scenario files should be placed in the `project/scenarios` directory, again using existing files as templates.
 
# Arguments

The main script accepts several command-line arguments to customize the training and video saving processes:

- `--train`: Start the training process.
- `--save`: Save a video of the agent's performance.
- `--model MODEL_NAME`: Specify the model architecture to use (default is `basic_cnn`).
- `--experiment EXPERIMENT_NAME`: Name of the experiment for logging purposes (default is `custom_doom_dm_explore`).
- `--seconds TRAINING_SECONDS`: Duration of training in seconds (default is `3600` seconds).
- `--workers NUM_WORKERS`: Number of parallel workers to use for training (default is `8`).
- `--worker-envs NUM_ENVIRONMENTS`: Number of environments per worker (default is `4`).

# Environment Variables

You can also set the following environment variables to customize the training and video saving processes:

- `EXPERIMENT`: Name of the experiment for logging purposes (default is `custom_doom_dm_explore`).
- `TRAINING_SECONDS`: Duration of training in seconds (default is `3600` seconds).
- `WORKERS`: Number of parallel workers to use for training (default is `8`).
- `WORKER_ENVS`: Number of environments per worker (default is `4`).
