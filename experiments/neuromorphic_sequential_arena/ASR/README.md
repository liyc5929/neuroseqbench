# 🧠 Quick Start Guide

This repository is built on top of the [ESPnet](https://github.com/espnet/espnet) toolkit. It provides training pipelines for Spiking Neural Networks (SNNs) in Automatic Speech Recognition (ASR) tasks.

Follow the instructions below to **reproduce the results reported in our paper**.

---

## 📦 1. Clone the Repository and Prepare the Working Directory

Please first navigate to the target directory:

```bash
cd ./neuroseqbench/experiments/neuromorphic_sequential_arena/ASR/
```

Then clone ESPnet and prepare necessary components:
```bash
git clone https://github.com/espnet/espnet
cd espnet

# Clean unused directories and existing setup
rm -rf egs egs2 espnet2 espnet espnetez setup.py

# Copy modified components from the benchmark repo
cp -r ../egs2 .
cp -r ../espnet2 .
cp -r ../tools .
cp ../setup.py .
```

## 🛠️ 2. Set Up the Environment and Install Dependencies

Install the environment:
```bash
cd tools
./setup_miniforge.sh miniconda espnet 3.8
conda activate espnet
pip install editdistance
make
```
> ⚠️ Note: If you encounter the error of "No matching distribution found for torch" during make, please install PyTorch manually based on your CUDA version. For CUDA 12.1, run:
>```bash
>pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
>```

Install kenlm:
```bash
make kenlm.done
```

Install neuroseqbench:
```bash
cd /neuroseqbench/
pip install -e .
```

## 📁 3. Configure the Dataset Path

Set the path to your AISHELL-1 dataset in:
```bash
egs2/aishell/asr1/db.sh
```

Edit the following line to point to your dataset location:
```bash
AISHELL=/your_path_to_aishell
```

## 🚀 4. Train a Spiking Neural Network Model
To launch training of the SNN model with LIF neurons, run:
```bash
cd egs2/aishell/asr1
bash run_snn_lif.sh
```
