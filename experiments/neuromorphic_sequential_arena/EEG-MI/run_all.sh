#!/bin/bash

# Feedforward Spiking Neuron
python ./main_train_EEG.py --lr 1e-3 --decay 1.0 --alpha 0.6 --neuron lif 
python ./main_train_EEG.py --lr 1e-3 --decay 0.3 --alpha 0.6 --neuron celif --hidden-dim 128 128 128
python ./main_train_EEG.py --lr 1e-3 --grad-clip 0.0 --decay 0.5 --alpha 0.6 --neuron ltc --hidden-dim 96 112
python ./main_train_EEG.py --lr 3e-3 --decay 0.5 --alpha 0.6 --neuron spsn --hidden-dim 128 256 256

# Recurrent Spiking Neuron
python ./main_train_EEG.py --lr 2e-4 --decay 1.0 --alpha 0.6 --neuron lif --recurrent --hidden-dim 128 176 176
python ./main_train_EEG.py --lr 2e-4 --decay 0.2 --alpha 0.6 --neuron celif --hidden-dim 64 128 128 --recurrent
python ./main_train_EEG.py --lr 1e-3 --grad-clip 0. --decay 0.5 --alpha 0.6 --neuron ltc --hidden-dim 96 104 --recurrent


# Neural Architecture
python ./main_train_EEG.py --lr 5e-3 --decay 0.5 --alpha 0.6 --neuron lif --net gsu --hidden-dim 64 112 112
python ./main_train_EEG.py --lr 3e-3 --decay 0.5 --alpha 1 --neuron lif --net tcn --ksize 7 --hidden-dim 28 28 28 28 28 28 28 28 --name spktcn_
python ./main_train_EEG.py --lr 5e-4 --decay 0.5 --alpha 0.6 --neuron lif --net spktransformer --hidden-dim 64 64 --nhead 8 --name spktrans_
python ./main_train_EEG.py --lr 1e-2 --weight-decay 5e-3 --decay 0.5 --alpha 0.6 --net binaryssm --hidden-dim 128 128
python ./main_train_EEG.py --lr 1e-2 --weight-decay 5e-3 --decay 0.5 --alpha 0.6 --net gsussm --hidden-dim 152 152
