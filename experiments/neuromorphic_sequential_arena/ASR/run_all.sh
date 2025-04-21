#!/bin/bash

# Feedforward Spiking Neuron
sh run ./scripts/snn_lif.sh
sh run ./scripts/snn_celif.sh
sh run ./scripts/snn_ltc.sh
sh run ./scripts/snn_spsn.sh

# Recurrent Spiking Neuron
sh run ./scripts/snn_rlif.sh
sh run ./scripts/snn_rcelif.sh
sh run ./scripts/snn_rltc.sh

# Neural Architecture
sh run ./scripts/snn_gsu.sh
sh run ./scripts/snn_tcn.sh
sh run ./scripts/snn_spktransformer.sh
sh run ./scripts/snn_binaryssm.sh
sh run ./scripts/snn_gsussm.sh
