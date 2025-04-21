#!/bin/bash

# Feedforward Spiking Neuron
python ./main_train_AL.py --lr 5e-3 --decay 1.0 --alpha 0.4 --neuron lif
python ./main_train_AL.py --lr 3e-3 --decay 0.5 --alpha 0.4 --neuron celif --hidden-dim 128 152 152
python ./main_train_AL.py --lr 1e-3 --decay 1.0 --alpha 0.4 --neuron ltc --hidden-dim 96 128
python ./main_train_AL.py --lr 5e-3 --decay 0.5 --alpha 0.4 --neuron spsn --hidden-dim 128 256 256

# Recurrent Spiking Neuron
python ./main_train_AL.py --lr 1e-3 --decay 0.5 --alpha 0.4 --neuron lif --recurrent --hidden-dim 128 176 176
python ./main_train_AL.py --lr 1e-3 --recurrent --decay 0.3 --alpha 0.4 --neuron celif --hidden-dim 120 120 120 --beta 0.03
python ./main_train_AL.py --lr 2e-3 --recurrent --decay 0.5 --alpha 0.4 --neuron ltc --hidden-dim 96 112 --grad-clip 0.25

# Neural Architecture
python ./main_train_AL.py --lr 3e-3 --decay 0.5 --alpha 0.6 --neuron lif --net gsu --hidden-dim 64 112 112 --name gsu_
python ./main_train_AL.py --lr 5e-3 --decay 0.5 --alpha 1 --neuron lif --net tcn --ksize 7 --hidden-dim 36 36 36 36 36 36 --name spktcn_
python ./main_train_AL.py --lr 5e-4 --decay 0.5 --alpha 0.6 --neuron lif --net spktransformer --hidden-dim 64 64  --nhead 8 --name spktrans_
python ./main_train_AL.py --lr 3e-3 --weight-decay 3e-3 --decay 0.5 --alpha 0.6 --net binaryssm --hidden-dim 128 128 --name binaryssm_
python ./main_train_AL.py --lr 5e-3 --weight-decay 3e-3 --decay 0.5 --alpha 0.6 --net gsussm --hidden-dim 152 152 --name GSUSSM_
