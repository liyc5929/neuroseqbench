# Audio Denoising Task Setup

> A step-by-step guide to install dependencies and run SNN-based audio denoising experiments using Accelerate and AudioZen.

### 1. Install core Python packages
```sh
pip install tensorboard joblib matplotlib
```

### 2. (Optional) Install ffmpeg for MP3 support
If you have "mp3" format audio data in your dataset, install ffmpeg first.
```sh
apt install ffmpeg
```

### 3. Install project dependencies from `requirements.txt`
```sh
pip install -r requirements.txt
```

### 4. Install AudioZen in editable mode
```sh
pip install -e .
```

### 5. Configure Accelerate for multi-GPU usage

To run an experiment of a model, we need to configuration the GPU usage. Accelerate provides a CLI tool that unifies all launchers, so you only have to remember one command. To use it, run a quick configuration setup first on your machine and answer the questions:
```sh
accelerate config
```

### 6. Launch training with a given config

We then use the following command to train SNN models:
```sh
accelerate launch run.py -C baseline_m.toml -M train
```

### 7. Set dataset root paths in your `.toml` config

Edit the root fields under `[train_dataset.args]` and `[validate_dataset.args]` in your `.toml` config file (e.g., `baseline_m.toml`) to point to the actual location of your N-DNS dataset:
```toml
[train_dataset.args]
root = "/benchmark_data/AD/training_set/"

[validate_dataset.args]
root = "/benchmark_data/AD/validation_set/"
```
This ensures that the data loader can correctly locate your training and validation data during experiment execution.
