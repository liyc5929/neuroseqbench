cd recipes/intel_ndns/spiking_fullsubnet/

# SFNN
## LIF
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46529 run.py -C lif.toml
## LTC
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46529 run.py -C ltc.toml
## SPSN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46528 run.py -C spsn.toml
## PMSN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46526 run.py -C pmsn.toml

# SRNN
## LIF
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46530 run.py -C rlif.toml
## LTC
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46534 run.py -C rltc.toml


# neural architecture
# GSU
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C gsn.toml
# TCN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C tcn.toml
# Spike-Driven Transformer
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C spikedriven_transformer.toml
# Binary S4D
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C binarys4d.toml
# GSU-SSM
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C gsussm.toml

