#!/bin/bash

cd ./egs2/aishell/asr1

# Feedforward Spiking Neuron
sh run snn_lif.sh
sh run snn_celif.sh
sh run snn_ltc.sh
sh run snn_spsn.sh

# Recurrent Spiking Neuron
sh run snn_rlif.sh
sh run snn_rcelif.sh
sh run snn_rltc.sh

# Neural Architecture
sh run snn_gsn.sh
sh run snn_tcn.sh
sh run snn_spktransformer.sh
sh run snn_binaryssm.sh
sh run snn_gsussm.sh
