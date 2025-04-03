#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/../.."

# PennTreeBank
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PTB_LIFwithDCLSDelays --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PTB_GSU --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PTB_TCN --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PTB_SpikingTCN --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PTB_LSTM --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PTB_Transformer --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PTB_SpikeDrivenTransformer --data_root /benchmark_data --device 0

# PSMNIST
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PSMNIST_LIFwithDCLSDelays --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PSMNIST_GSU --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PSMNIST_TCN --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PSMNIST_SpikingTCN --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PSMNIST_LSTM --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PSMNIST_Transformer --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item PSMNIST_SpikeDrivenTransformer --data_root /benchmark_data --device 0

# Binary Adding
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_LIF --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_LIFwithDCLSDelays --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_GSU --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_TCN --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_SpikingTCN --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_LSTM --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_Transformer --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 06_neuron_arch_on_benchmarks --experiment_item BinaryAdding_SpikeDrivenTransformer --data_root /benchmark_data --device 0
