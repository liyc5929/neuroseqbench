#!/bin/bash

# Feedforward Spiking Neuron
python ./main_train_HAR.py --lr 3e-3 --decay 0.9 --alpha 0.6 --neuron lif
python ./main_train_HAR.py --lr 3e-3 --decay 0.3 --alpha 0.6 --neuron celif --hidden-dim 128 192 192 --beta 0.15
python ./main_train_HAR.py --lr 3e-3 --decay 0.5 --alpha 0.6 --neuron ltc --hidden-dim 96 128
python ./main_train_HAR.py --lr 3e-3 --decay 0.5 --alpha 0.6 --neuron spsn --hidden-dim 128 256 256
python ./main_train_HAR.py --lr 5e-3 --decay 0.5 --neuron pmsn --hidden-dim 64 256 256

# Recurrent Spiking Neuron
python ./main_train_HAR.py --lr 1.5e-3 --grad-clip 0. --decay 0.1 --alpha 0.6 --neuron lif --recurrent --hidden-dim 128 176 176
python ./main_train_HAR.py --lr 3e-3  --decay 0.3 --alpha 0.6 --neuron celif --hidden-dim 128 128 160 --recurrent --beta 0.1
python ./main_train_HAR.py --lr 3e-3  --decay 0.5 --alpha 0.6 --neuron ltc --hidden-dim 96 112 --recurrent

# Neural Architecture
python ./main_train_HAR.py --lr 5e-3 --decay 0.5 --alpha 0.6 --neuron lif --net gsu --hidden-dim 64 112 112
python ./main_train_HAR.py --lr 5e-3 --decay 0.5 --alpha 1 --neuron lif --net tcn --ksize 16 --hidden-dim 76 --name spktcn
python ./main_train_HAR.py --lr 3e-3 --decay 0.5 --alpha 0.8 --neuron lif --net spktransformer --hidden-dim 64 64 --name spktrans --nhead 1 --threshold 0.6
python ./main_train_HAR.py --lr 1e-2 --weight-decay 3e-3 --decay 0.5 --alpha 0.6 --net binaryssm --hidden-dim 128 128
python ./main_train_HAR.py --lr 1e-2 --weight-decay 3e-3 --decay 0.5 --alpha 0.6 --net gsussm --hidden-dim 152 152
