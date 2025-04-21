#!/bin/bash

# Feedforward Spiking Neuron
python ./main_train_SSL.py --lr 5e-3 --decay 0.6 --alpha 0.6 --neuron lif
python ./main_train_SSL.py --lr 5e-3 --decay 0.3 --alpha 0.6 --neuron celif --hidden-dim 96 128 128 --beta 0.05
python ./main_train_SSL.py --lr 3e-3 --decay 0.5 --alpha 0.6 --neuron ltc --hidden-dim 96 112 --grad-clip 1.0
python ./main_train_SSL.py --lr 3e-3 --decay 0.5 --alpha 0.6 --neuron spsn --hidden-dim 128 256 256

# Recurrent Spiking Neuron
python ./main_train_SSL.py --lr 5e-4 --decay 0.5 --alpha 0.4 --neuron lif --recurrent --hidden-dim 128 168 168
python ./main_train_SSL.py --lr 1e-3 --decay 0.3 --alpha 0.6 --neuron celif --hidden-dim 96 108 108 --recurrent
python ./main_train_SSL.py --lr 1e-3 --grad-clip 0.5 --decay 0.5 --alpha 0.6 --neuron ltc --hidden-dim 96 104 --recurrent

# Neural Architecture
python ./main_train_SSL.py --lr 1e-3 --decay 0.5 --alpha 0.4 --neuron lif --net gsu --hidden-dim 64 112 112 --name gsu_ --grad-clip 0.15
python ./main_train_SSL.py --lr 3e-3 --decay 0.5 --alpha 1 --neuron lif --net tcn --ksize 3 --hidden-dim 46 46 46 46 46 46 46 46
python ./main_train_SSL.py --lr 5e-4 --decay 0.5 --alpha 0.6 --neuron lif --net spktransformer --hidden-dim 64 64 --nhead 4
python ./main_train_SSL.py --lr 1e-2 --weight-decay 5e-3 --decay 0.5 --alpha 0.6 --net binaryssm --hidden-dim 128 128
python ./main_train_SSL.py --lr 1e-2 --weight-decay 5e-3 --decay 0.5 --alpha 0.6 --net gsussm --hidden-dim 152 152
