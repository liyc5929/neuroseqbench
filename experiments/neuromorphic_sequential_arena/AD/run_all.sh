#!/bin/bash

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# SFNN
# ## LIF
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46529 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/lif.toml
## LTC
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46529 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/ltc.toml
## SPSN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46528 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/spsn.toml
# PMSN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46526 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/pmsn.toml


# SRNN
## LIF
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46530 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/rlif.toml
## LTC
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46534 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/rltc.toml


# neural architecture
# GSN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/gsn.toml
# TCN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/tcn.toml
# Spike-Driven Transformer
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/spikedriven_transformer.toml
# Binary S4D
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/binarys4d.toml
# GSU-SSM
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 "${TASK_DIR}/run.py" -C ${TASK_DIR}/conf/gsussm.toml
