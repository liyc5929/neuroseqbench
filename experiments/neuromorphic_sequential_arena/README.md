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
    ```sh
    wget [cloud_drive_link] -O data/AL.zip
    unzip data/AL.zip -d data/AL/
    ```

  - **Option 2: Automatic Dataset Generation**  
    If the dataset is not found in the directory, our **dataloader will automatically generate a new one**. However, this process may take a significant amount of time. Additionally, due to variations in the generated data, there may be **differences in reproducibility**. Users who wish to ensure consistency should download the pre-generated dataset instead.  
    ```sh
    python main_AL.py --data_path data
    ```

### 2. Human Activities Recognition (HAR)
- **Dataset:** WISDM dataset
- **Source:** [WISDM Smartphone and Smartwatch Activity Smartphone and Smartwatch Activity and and
Biometrics Dataset Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00507/WISDM-dataset-description.pdf)
- **Dataset Access:**
  - **🔹 Option 1: Download Pre-Processed Dataset (Recommended)**
  
    Users can directly download the pre-processed dataset, which includes four `.npy` files corresponding to the training/testing data and labels. 
    ```sh
    wget [cloud_drive_link] -O data/HAR.zip
    unzip data/HAR.zip -d data/HAR/
    ```
  
  - **Option 2: Download Raw WISDM Dataset and Perform Preprocessing Yourself**
  
    Alternatively, users can download the **raw WISDM dataset** from the [official source](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+sma%20rtwatch+activity+and+biometrics+dataset) and unzip it under `data/HAR`. 
    ```sh
    unzip data/HAR/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip -d data/HAR/
    unzip data/HAR/wisdm-dataset.zip -d data/HAR/
    ```
    - **Preprocessing:**
  
      Our dataloader will automatically handle preprocessing and automatically convert it into the required `.npy` format:
      ```sh
      python main_HAR.py --data_path data
      ```

### 3. Electroencephalogram Motor Imagery (EEG-MI)
- **Dataset:** OpenBMI dataset  
- **Source:**  [EEG Dataset and OpenBMI Toolbox for Three BCI Paradigms: An Investigation into BCI Illiteracy](https://gigadb.org/dataset/100542)  
- **Dataset Access:**  
  - **🔹 Option 1: Download Pre-Processed Dataset (Recommended)**
  
    Users can directly download the pre-processed dataset, which includes four `.npy` files corresponding to the training/testing data and labels.  
    ```sh
    wget [cloud_drive_link] -O data/EEG.zip
    unzip data/EEG.zip -d data/EEG/
    ```
  
  - **Option 2: Download Raw OpenBMI Dataset and Perform Preprocessing Yourself**
  
    Alternatively, users can download all files ending with `_MI.mat` from the **OpenBMI** [official source](https://gigadb.org/dataset/view/id/100542/File_page/5/Files_page/23) and place them under `data/EEG/eeg_data`. 
    - **Preprocessing:**  
      1. Use our provided **MATLAB preprocessing script** to convert them into `data_*.mat` and `label_*.mat` files (108 in total).  
      2. Our dataloader will automatically read and convert the processed `.mat` files into the required `.npy` format:  
      ```sh
      python main_EEG.py --data_path data
      ```

### 4. Sound Source Localization (SSL)
- **Dataset:** SLoClas dataset 
- **Source:** [SLoClas: A Database for Joint Sound Localization and Classification](https://ieeexplore.ieee.org/document/9660517)
- **Dataset Access:**  
  - **Option 1: Download Pre-Processed Dataset (Recommended)**  
    Users can directly download the pre-processed dataset, which includes  `training_raw_noise.mat` and `testing_raw_noise.mat`.
    ```sh
    wget [cloud_drive_link] -O data/SSL.zip
    unzip data/SSL.zip -d data/SSL/
    ```
  
  - **Option 2: Download Raw SLoClas Dataset and Perform Preprocessing Yourself**  
    Alternatively, users can download the raw **SLoClas dataset** (`.wav` files) from the [official source](https://zenodo.org/records/5211296) and place it under `data/SSL/ssl_data`.
    ```sh
    unzip data/SSL/ssl_data/SoClas_database.zip -d data/SSL/ssl_data
    ```
    - **Preprocessing:**  
      1. Use our provided **MATLAB preprocessing script** to segment samples and add noise. After this step, you will obtain `training_raw_noise.mat` and `testing_raw_noise.mat`, which represent the preprocessed training and testing sets.  
      2. Our dataloader will then automatically load these preprocessed `.mat` files for model training.
          ```sh
          python main_SSL.py --data_path data
          ```

### 5. Automatic Lip-Reading (ALR)

- **Dataset:** DVS-Lip dataset  
- **Source:** [Multi-grained Spatio-Temporal Features Perceived Network for Event-based Lip-Reading](https://ieeexplore.ieee.org/document/9879993)
- **Dataset Access:** 
  - **Download Pre-Processed Dataset (Recommended)**  
      Users can directly download [the DVS-Lip dataset](https://drive.google.com/file/d/1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0/view) by the following steps:
      ```sh
      wget https://drive.usercontent.google.com/download?id=1dBEgtmctTTWJlWnuWxFtk8gfOdVVpkQ0&export=download&authuser=0&confirm=t&uuid=005c1c2f-2ada-4975-9d2f-c603853d850e&at=AEz70l7bjhEqid1uPWEc2AoiYQC-%3A1742207401385 -O data/ALR.zip
      unzip data/ALR.zip -d data/ALR/
      ```

### 6. Audio Denoising (AD)
- **Dataset:** N-DNS  
- **Source:** [The Intel neuromorphic DNS challenge](https://iopscience.iop.org/article/10.1088/2634-4386/ace737)
- **Dataset Access:**
  - **Option 1: Download Pre-Processed Dataset (Recommended)**  
      Users can directly download the N-DNS dataset by the following steps:
      ```sh
      wget -- -O data/AD.zip
      unzip data/AD.zip -d data/AD/
      ```
  - **Option 2: Please refer to [Intel Neuromorphic DNS Challenge Datasets](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge#dataset) for preparing the dataset**

### 7. Automatic Speech Recognition (ASR)
- **Dataset:** AISHELL  
- **Source:** [AISHELL-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline](https://arxiv.org/abs/1709.05522)
- **Dataset Access:** 
  - This dataset will be automatically downloaded when training SNN models.  


---

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

---

## Notes

- Ensure that all datasets are **downloaded and preprocessed before running experiments**.  
- If you encounter issues with downloading or preprocessing, first ensure that the dataset is placed in the correct directory as specified above. If the issue persists, check the dataset source pages for additional instructions. 
---



