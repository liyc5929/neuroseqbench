#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/../.."

# PennTreeBank
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_PLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_PLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_ALIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_ALIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_GLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_GLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_CLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_CLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_CELIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_CELIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_TCLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_TCLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LMH_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LMH_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_adLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_adLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LTC_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LTC_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_DHLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_DHLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_SPSN_feedforward --data_root /benchmark_data --device 0


# PSMNIST
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_PLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_PLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_ALIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_ALIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_adLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_adLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_GLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_GLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LTC_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LTC_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_TCLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_TCLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LMH_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LMH_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_CLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_CLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_DHLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_DHLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_CELIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_CELIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_SPSN_feedforward --data_root /benchmark_data --device 0


# Binary Adding
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_PLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_PLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_ALIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_ALIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_adLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_adLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_GLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_GLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LTC_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LTC_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_TCLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_TCLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LMH_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LMH_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_CLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_CLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_DHLIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_DHLIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_CELIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_CELIF_recurrent --data_root /benchmark_data --device 0

python ./experiments/runner.py --paper_name segregated_temporal_probe --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_SPSN_feedforward --data_root /benchmark_data --device 0
