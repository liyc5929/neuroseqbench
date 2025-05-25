# Data Preparation

This file provides instructions on how to obtain and prepare the datasets used in **Neuromorphic Sequential Arena (NSA)**, a benchmark for neuromorphic temporal processing. NSA includes **seven tasks**, each with a different dataset and acquisition method.

## Available Tasks

The NSA benchmark consists of the following tasks:

1. **Autonomous Localization (AL)**
2. **Human Activities Recognition (HAR)**
3. **Electroencephalogram Motor Imagery (EEG-MI)**
4. **Sound Source Localization (SSL)**
5. **Automatic Lip-Reading (ALR)**
6. **Audio Denoising (AD)**
7. **Automatic Speech Recognition (ASR)**

Each task requires specific datasets, which can be downloaded and processed using the instructions below.

---

## Downloading and Preparing Datasets

### 1. Autonomous Localization (AL)
- **Dataset:** AL dataset (Synthetic)
- **Dataset Access:**  
  - 🔹 **Option 1： Download Pre-Generated Dataset (Recommended)**  
    Users can directly download the pre-generated dataset and place it in the designated data directory for immediate use.

    Download the dataset (`AL.zip`) from [our Hugging Face repository](https://huggingface.co/datasets/liyc5929/neuroseqbench/tree/main/neuromorphic_sequential_arena/AL), then extract it with the following command:

    ```shell
    unzip AL.zip -d benchmark_data/AL/
    ```

  - **Option 2: Automatic Dataset Generation**  
    If the dataset is not found in the directory, our **dataloader will automatically generate a new one**. However, this process may take a significant amount of time. Additionally, due to variations in the generated data, there may be **differences in reproducibility**. Users who wish to ensure consistency should download the pre-generated dataset instead.  
    ```shell
    python ./AL/main.py --data_path benchmark_data/
    ```

### 2. Human Activities Recognition (HAR)
- **Dataset:** WISDM dataset
- **Source:** [WISDM Smartphone and Smartwatch Activity Smartphone and Smartwatch Activity and and
Biometrics Dataset Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-dataset-description.pdf)
- **Dataset Access:**
  - **🔹 Option 1: Download Pre-Processed Dataset (Recommended)**
  
    Users can directly download the pre-processed dataset, which includes four `.npy` files corresponding to the training/testing data and labels. 
    Download the dataset (`WISDM.zip`) from [our Hugging Face repository](https://huggingface.co/datasets/liyc5929/neuroseqbench/tree/main/neuromorphic_sequential_arena/WISDM), then extract it with the following command:
    ```shell
    unzip WISDM.zip -d benchmark_data/HAR/
    ```
  
  - **Option 2: Download Raw WISDM Dataset and Perform Preprocessing Yourself**
  
    Alternatively, users can download the **raw WISDM dataset** from the [official source](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+sma%20rtwatch+activity+and+biometrics+dataset) and unzip it under `benchmark_data/HAR`. 
    ```shell
    unzip wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip -d benchmark_data/HAR/
    unzip wisdm-dataset.zip -d benchmark_data/HAR/
    ```
    - **Preprocessing:**
  
      Our dataloader will automatically handle preprocessing and convert it into the required `.npy` format:
      ```shell
      python ./HAR/main.py --data_path benchmark_data/
      ```

### 3. Electroencephalogram Motor Imagery (EEG-MI)
- **Dataset:** OpenBMI dataset  
- **Source:**  [EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy](https://gigadb.org/dataset/100542)  
- **Dataset Access:**  
  - **🔹 Option 1: Download Pre-Processed Dataset (Recommended)**
  
    Users can directly download the pre-processed dataset, which includes four `.npy` files corresponding to the training/testing data and labels.
    Download the dataset (`OpenBMI.zip`) from [our Hugging Face repository](https://huggingface.co/datasets/liyc5929/neuroseqbench/tree/main/neuromorphic_sequential_arena/OpenBMI), then extract it with the following command:
    ```shell
    unzip OpenBMI.zip -d benchmark_data/EEG/
    ```
  
  - **Option 2: Download Raw OpenBMI Dataset and Perform Preprocessing Yourself**
  
    Alternatively, users can download all files ending with `_MI.mat` from the **OpenBMI** [official source](https://gigadb.org/dataset/view/id/100542/File_page/5/Files_page/23) and place them under `benchmark_data/EEG/eeg_data`. 
    - **Preprocessing:**  
      1. Use our provided [**MATLAB preprocessing scripts**](./EEG-MI/preprocess_matlab/) to convert them into `data_*.mat` and `label_*.mat` files (108 in total).  
      2. Our dataloader will automatically read and process all `.mat` files into the required `.npy` format:  
      ```shell
      python ./EEG-MI/main.py --data_path benchmark_data/
      ```

### 4. Sound Source Localization (SSL)
- **Dataset:** SLoClas dataset 
- **Source:** [SLoClas: A Database for Joint Sound Localization and Classification](https://ieeexplore.ieee.org/document/9660517)
- **Dataset Access:**  
  - **Option 1: Download Pre-Processed Dataset (Recommended)**  
    Users can directly download the pre-processed dataset, which includes  `training_raw_noise.mat` and `testing_raw_noise.mat`.
    Download the dataset (`SLoClas.zip`) from [our Hugging Face repository](https://huggingface.co/datasets/liyc5929/neuroseqbench/tree/main/neuromorphic_sequential_arena/SLoClas), then extract it with the following command:
    ```shell
    unzip SLoClas.zip -d benchmark_data/SSL/
    ```
  
  - **Option 2: Download Raw SLoClas Dataset and Perform Preprocessing Yourself**  
    Alternatively, users can download the raw **SLoClas dataset** (`.wav` files) from the [official source](https://zenodo.org/records/5211296) and unzip it.
    ```shell
    unzip SoClas_database.zip
    ```
    - **Preprocessing:**  
      1. Use our provided [**MATLAB preprocessing scripts**](./SSL/preprocess_matlab) to segment samples and add noise. This will generate two files: `training_raw_noise.mat` and `testing_raw_noise.mat`, which represent the preprocessed training and testing sets, respectively.
      2. Place both `.mat` files in the `benchmark_data/SSL/` directory.
      3. Our dataloader will then automatically load these preprocessed `.mat` files for model training.
          ```shell
          python ./SSL/main.py --data_path benchmark_data/
          ```

### 5. Automatic Lip-Reading (ALR)

- **Dataset:** DVS-Lip dataset  
- **Source:** [Multi-grained Spatio-Temporal Features Perceived Network for Event-based Lip-Reading](https://ieeexplore.ieee.org/document/9879993)
- **Dataset Access:** 
  - **Download Pre-Processed Dataset (Recommended)**  
      Users can directly download [the DVS-Lip dataset](https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view) by the following steps:
      ```shell
      wget https://drive.usercontent.google.com/download?id=1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0&export=download&authuser=0&confirm=t&uuid=005c1c2f-2ada-4975-9d2f-c603853d850e&at=AEz70l7bjhEqid1uPWEc2AoiYQC-%3A1742207401385 -O ALR.zip
      unzip ALR.zip -d benchmark_data/ALR/
      ```

### 6. Audio Denoising (AD)
- **Dataset:** N-DNS  
- **Source:** [The Intel neuromorphic DNS challenge](https://iopscience.iop.org/article/10.1088/2634-4386/ace737)
- **Dataset Access:**
  - **Option 1: Download Pre-Processed Dataset (Recommended)**  
      Download the dataset (`N-DNS.zip`) from [our Hugging Face repository](https://huggingface.co/datasets/liyc5929/neuroseqbench/tree/main/neuromorphic_sequential_arena/N-DNS), then extract it with the following command:
      ```shell
      unzip N-DNS.zip -d benchmark_data/AD/
      ```
  - **Option 2: Please refer to [Intel Neuromorphic DNS Challenge Datasets](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#dataset) for preparing the dataset**

### 7. Automatic Speech Recognition (ASR)
- **Dataset:** AISHELL  
- **Source:** [AISHELL-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline](https://arxiv.org/abs/1709.05522)
- **Dataset Access:** 
  - This dataset will be automatically downloaded when training SNN models.  


## Dataset Structure

Each dataset should be organized in the following structure under `data/` after downloading and preprocessing:

```
data/
│── AL/                  # Autonomous Localization task
│── HAR/                 # Human Activities Recognition task
│── EEG/                 # Electroencephalogram Motor Imagery task
│── SSL/                 # Sound Source Localization task
│── ALR/                 # Automatic Lip-Reading task
│── AD/                  # Audio Denoising task
│── ASR/                 # Automatic Speech Recognition task
```

## Notes

- Ensure that all datasets are **downloaded and preprocessed before running experiments**.  
- If you encounter issues with downloading or preprocessing, first ensure that the dataset is placed in the correct directory as specified above. If the issue persists, check the dataset source pages for additional instructions. 
---



