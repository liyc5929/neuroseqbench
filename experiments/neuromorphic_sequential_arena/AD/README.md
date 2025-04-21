


1. Install Conda dependencies
```sh
conda install tensorboard joblib matplotlib
```

2. (Optional) If you have "mp3" format audio data in your dataset, install ffmpeg first.
```sh
conda install ffmpeg -c conda-forge
```

3. Install PyPI dependencies 
```sh
pip install -r requirements.txt
```

4. Install the audiozen package in editable mode.
```sh
pip install -e .
```

5. To run an experiment of a model, we first go to a model directory.
```sh
cd recipes/intel_ndns/spiking_fullsubnet/
```

6. First, we need to configuration the GPU usage. Accelerate provides a CLI tool that unifies all launchers, so you only have to remember one command. To use it, run a quick configuration setup first on your machine and answer the questions:
```sh
accelerate config
```

7. Then, we can use the following command to train SNN models
```sh
accelerate launch run.py -C baseline_m.toml -M train
```