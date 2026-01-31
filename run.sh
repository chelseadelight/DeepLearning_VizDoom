#!/bin/sh

export EXPERIMENT="gather_then_fight3"
export SCENARIO="custom_b7a1_dm"
export WORKERS=16
export WORKER_ENVS=16
export ARCHITECTURE="gru"

python -m project.main $@
