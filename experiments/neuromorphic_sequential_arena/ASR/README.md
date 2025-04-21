conda env create -f=./espnet_env.yaml -p ~/env_espnet

cd espnet

pip install -e .

conda activate ~/env_espnet

make kenlm.done

pip install espnet_tts_frontend

cd ./espnet/egs2/aishell/asr1

bash run_snn_lif.sh