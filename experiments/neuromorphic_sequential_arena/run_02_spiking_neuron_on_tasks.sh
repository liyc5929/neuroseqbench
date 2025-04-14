#!/bin/bash

# Navigate to the project root directory
cd "$(dirname "$0")/../.."

# Autonomous localization
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item AL_LIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item AL_LIF_recurrent --data_root /benchmark_data --device 0

# Human activities recognition
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item HAR_LIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item HAR_LIF_recurrent --data_root /benchmark_data --device 0

# Electroencephalogram motor imagery
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item EEG-MI_LIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item EEG-MI_LIF_recurrent --data_root /benchmark_data --device 0

# Sound source localization
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item SSL_LIF_feedforward --data_root /benchmark_data --device 0
python ./experiments/runner.py --paper_name neuromorphic_sequential_arena --experiment_name 02_spiking_neuron_on_tasks --experiment_item SSL_LIF_recurrent --data_root /benchmark_data --device 0
