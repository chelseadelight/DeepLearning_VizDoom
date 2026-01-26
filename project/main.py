import functools
import argparse

from multiprocessing import freeze_support
import os

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sample_factory.enjoy import enjoy

from sf_examples.vizdoom.doom.doom_model import make_vizdoom_encoder
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom.doom.doom_utils import ADDITIONAL_INPUT, BOTS_REWARD_SHAPING, DOOM_ENVS, DoomSpec, doom_action_space_full_discretized, make_doom_env_from_spec
from sf_examples.vizdoom.doom.wrappers.exploration import ExplorationWrapper

from .models.custom_encoder import make_custom_vizdoom_encoder

# Helper function to get absolute path to scenario files
def abs_scenario(scenario_file):
    """Get absolute path to a scenario file located in the 'scenarios' directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_dir = os.path.join(current_dir, 'scenarios')
    return os.path.join(scenarios_dir, scenario_file)

# Exploration wrapper configuration
EXPLORATION_REWARD = (
    ExplorationWrapper,
    {}
)

# Custom ViZDoom environments with exploration and reward shaping
CUSTOM_ENVS = [
    DoomSpec(
        "custom_doom_dm_explore",
        abs_scenario("dm_custom.cfg"),
        doom_action_space_full_discretized(),
        1.0,
        int(1e9),
        num_agents=1,
        num_bots=7,
        extra_wrappers=[ADDITIONAL_INPUT, EXPLORATION_REWARD, BOTS_REWARD_SHAPING],
    )
]

# List of available model architectures
MODELS = [
    ("default", make_vizdoom_encoder),
    ("custom", make_custom_vizdoom_encoder),
]

# Registers all the ViZDoom environments
def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)

    for env_spec in CUSTOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


# Sample Factory allows the registration of a custom Neural Network architecture
# See https://github.com/alex-petrenko/sample-factory/blob/master/sf_examples/vizdoom/doom/doom_model.py for more details
def register_vizdoom_models(model_name="default"):
    for name, factory in MODELS:
        if name == model_name:
            global_model_factory().register_encoder_factory(factory)
            return
    # If the specified model_name is not found, register the default model
    global_model_factory().register_encoder_factory(make_vizdoom_encoder)



# parse the command line args and create a config
def parse_vizdoom_cfg(argv=None, evaluation=False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def intOrNone(value):
    try:
        return int(value)
    except:
        return None

def envOrVal(var_name, default_value):
    value = os.getenv(var_name)
    if value is None:
        return default_value
    return value

if __name__ == '__main__':
    # accept arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--save', action='store_true', help='Save the model and generate videos')
    parser.add_argument('--use-cpu', action='store_true', help='Use cpu only')
    parser.add_argument('--experiment', type=str, default=envOrVal("EXPERIMENT", "default"), help='Experiment name')
    parser.add_argument('--seconds', type=int, default=envOrVal("TRAINING_SECONDS", 3600), help='Training duration in seconds')
    parser.add_argument('--workers', type=int, default=envOrVal("WORKERS", 8), help='Number of workers')
    parser.add_argument('--worker-envs', type=int, default=envOrVal("WORKER_ENVS", 4), help='Number of environments per worker')
    parser.add_argument('--model', type=str, default=envOrVal("MODEL", "default"), help='Model architecture to use (default or custom)')
    parser.add_argument('--architecture', type=str, default=envOrVal("ARCHITECTURE", "baseline"), choices=["baseline", "gru"], help='Architecture to use')
    args = parser.parse_args()


    # Required for Windows multiprocessing
    freeze_support() 

    # Register ViZDoom environments and models
    register_vizdoom_envs()
    register_vizdoom_models(model_name=args.model)

    env = "custom_doom_dm_explore"
    exp = args.experiment
    if exp == "default":
        exp = env

    exp = f"{exp}_{args.model}_{args.architecture}"

    if args.train:
        argv=[
            f"--env={env}",
            f"--experiment={exp}",
            f"--num_workers={args.workers}",
            f"--num_envs_per_worker={args.worker_envs}",
            f"--train_for_seconds={args.seconds}",
            "--env_frameskip=4",
            "--num_policies=1",
        ]

        # allows checking training behavior without CUDA
        # not recommended to be used for full training runs
        if args.use_cpu:
            argv += [
                "--device=cpu"
            ]
        else:
            argv += [
                "--device=gpu"
            ]

        if args.architecture == "gru":
            argv += [
                "--use_rnn=True",
                "--rnn_type=gru",
                "--rnn_size=256",
                "--rnn_num_layers=1",
            ]

        cfg = parse_vizdoom_cfg(argv=argv)
        status = run_rl(cfg)

    if args.save:
        argv=[
            f"--env={env}",
            f"--experiment={exp}",
            "--num_workers=1",
            "--save_video",
            "--no_render",
            "--max_num_episodes=10"
        ]

        # allows checking evaluation behavior without CUDA
        # not recommended to be used for full evaluation runs
        if args.use_cpu:
            argv += [
                "--device=cpu"
            ]

        cfg = parse_vizdoom_cfg(argv=argv, evaluation=True)
        status = enjoy(cfg)
