#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/../.."

# PennTreeBank
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PTB_TEBN_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PTB_TEBN_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PTB_TDBN_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PTB_TDBN_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PTB_LayerNorm_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PTB_LayerNorm_recurrent --data_root /benchmark_data --device 0

# PSMNIST
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PSMNIST_TEBN_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PSMNIST_TEBN_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PSMNIST_TDBN_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PSMNIST_TDBN_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PSMNIST_LayerNorm_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item PSMNIST_LayerNorm_recurrent --data_root /benchmark_data --device 0

# Binary Adding
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item BinaryAdding_TEBN_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item BinaryAdding_TEBN_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item BinaryAdding_TDBN_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item BinaryAdding_TDBN_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item BinaryAdding_LayerNorm_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 04_normalization_on_benchmarks --experiment_item BinaryAdding_LayerNorm_recurrent --data_root /benchmark_data --device 0
