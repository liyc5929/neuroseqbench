
# SFNN
## LIF
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46529 run.py -C ./conf/lif.toml
## LTC
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46529 run.py -C ./conf/ltc.toml
## SPSN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46528 run.py -C ./conf/spsn.toml
## PMSN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46526 run.py -C ./conf/pmsn.toml

# SRNN
## LIF
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46530 run.py -C ./conf/rlif.toml
## LTC
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46534 run.py -C ./conf/rltc.toml


# neural architecture
# GSU
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C ./conf/gsn.toml
# TCN
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C ./conf/tcn.toml
# Spike-Driven Transformer
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C ./conf/spikedriven_transformer.toml
# Binary S4D
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C ./conf/binarys4d.toml
# GSU-SSM
accelerate launch --multi_gpu --num_processes=2 --gpu_ids 0,1 --main_process_port 46531 run.py -C ./conf/gsussm.toml

