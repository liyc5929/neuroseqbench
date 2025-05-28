#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/../.."

# PennTreeBank
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item PTB_STBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item PTB_SDBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item PTB_NoTD --data_root /benchmark_data --device 0

# PSMNIST
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item PSMNIST_STBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item PSMNIST_SDBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item PSMNIST_NoTD --data_root /benchmark_data --device 0

# Binary Adding
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item BinaryAdding_STBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item BinaryAdding_SDBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item BinaryAdding_NoTD --data_root /benchmark_data --device 0


# Spiking Heidelberg Digits
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item SHD_STBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item SHD_SDBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item SHD_NoTD --data_root /benchmark_data --device 0

# Spiking Speech Commands
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item SSC_STBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item SSC_SDBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item SSC_NoTD --data_root /benchmark_data --device 0

# TIMIT
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item TIMIT_STBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item TIMIT_SDBP --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 01_STP_on_benchmarks --experiment_item TIMIT_NoTD --data_root /benchmark_data --device 0
