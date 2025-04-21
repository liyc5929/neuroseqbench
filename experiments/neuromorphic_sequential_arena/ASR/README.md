```shell
conda env create -f=./espnet_env.yaml -p ~/env_espnet

cd espnet

pip install -e .

conda activate ~/env_espnet

make kenlm.done

pip install espnet_tts_frontend

bash run_all.sh
```
