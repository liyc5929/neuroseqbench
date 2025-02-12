# Neuromorphic Sequential Benchmark

The goal of Neuromorphic Sequential Benchmark is to enable consistent performance comparisons across different approaches to Spiking Neural Networks (SNNs) for temporal processing and to facilitate the tracking of advancements in the field.

This repository currently contains the source code and implementation details for the evaluation results in the research paper titled ["Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects"](link). We are in the process of developing a comprehensive benchmark suite tailored for neuromorphic temporal processing. Guidelines will be provided to guarantee fair and consistent evaluations of emerging SNN approaches using this benchmark suite.

We warmly invite researchers and practitioners in the field of neuromorphic temporal processing to engage with us by providing feedback and contributing. By integrating more comprehensive temporal processing benchmarks and advanced SNN methods, your contributions can significantly advance this field. We value your insights and look forward to collaborating to drive innovation together.



---

<h3 align="center"> Spiking Neural Networks for Temporal Processing: Status Quo and Future Prospects </h3>

---

> **Abstract:** Temporal processing is fundamental for both biological and artificial intelligence systems, as it enables the comprehension of dynamic environments and facilitates timely responses. Spiking Neural Networks (SNNs) excel in handling such data with high efficiency, owing to their rich neuronal dynamics and sparse activity patterns. Given the recent surge in the development of SNNs, there is an urgent need for a comprehensive evaluation of their temporal processing capabilities. In this paper, we first conduct an in-depth assessment of commonly used neuromorphic benchmarks, revealing critical limitations in their ability to evaluate the temporal processing capabilities of SNNs. To bridge this gap, we further introduce a benchmark suite consisting of three temporal processing tasks characterized by rich temporal dynamics across multiple timescales. Utilizing this benchmark suite, we perform a thorough evaluation of recently introduced SNN approaches to elucidate the current status of SNNs in temporal processing. Our findings indicate significant advancements in recently developed spiking neuron models and neural architectures regarding their temporal processing capabilities, while also highlighting a performance gap in handling long-range dependencies when compared to state-of-the-art non-spiking models. Finally, we discuss the key challenges and outline potential avenues for future research.

### Features

The following illustration depicts the Segregated Temporal Probe (STP), an analytical tool for assessing the effectiveness of neuromorphic benchmarks in evaluating the temporal processing capabilities of SNNs. The STP incorporates three algorithms—Spatio-Temporal Backpropagation (STBP), Spatial Domain Backpropagation (SDBP), and No Temporal Domain (NoTD)—which systematically disrupt the temporal processing pathways within an SNN to elucidate their significance.

<p align="center">
  <img src="./docs/_statics/overview.jpg" alt="STP overview" width="95%" />
</p>

The table below provides a detailed table outlining the key components of our framework. Each component is described with examples of instances and where they can be found within the repository. This information is intended to help users quickly understand the capabilities and structure of our system.

<div align="center">

| Components          |                   Description / Instances                    |      Repository Location      |
| ------------------- | :----------------------------------------------------------: | :---------------------------: |
| Neuron Model        |      LIF, ALIF, PLIF, GLIF, Normalization Layers, etc.       |  `framework/network/neuron`   |
| Neural Architecture | DCLS-Delays, SpikingTCN, Gated Spiking Neuron, Spike-Driven Transformer, etc. | `framework/network/structure` |
| Dataset             | Penn Treebank, Permuted Sequential MNIST, Binary Adding, etc. |   `framework/utils/dataset`   |

</div>

### Main Results

- Results for different **learning algorithms** across benchmark suites, "FF" and "Rec." refer to "feedforward" and "recurrent" architectures, respectively. "PPL" stands for "perplexity".

<table border="1" style="width: 100%; border-collapse: collapse;" align="center">
    <thead>
        <tr>
            <th>Dataset</th>
            <th colspan="2" style="text-align: center;">PTB ($T=70$)</th>
            <th colspan="2" style="text-align: center;">PS-MNIST ($T=784$)</th>
            <th colspan="2" style="text-align: center;">Binary Adding ($T=100$)</th>
        </tr>
        <tr>
            <th>Metric</th>
            <th style="text-align: center;">PPL $\downarrow$</th>
            <th style="text-align: center;">Rec.</th>
            <th style="text-align: center;">Acc. $\uparrow$</th>
            <th style="text-align: center;">Rec.</th>
            <th style="text-align: center;">Acc. $\uparrow$</th>
            <th style="text-align: center;">Rec.</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>STBP</td>
            <td style="text-align: center;">129.96</td>
            <td style="text-align: center;">111.96</td>
            <td style="text-align: center;">57.45</td>
            <td style="text-align: center;">72.97</td>
            <td style="text-align: center;">29.60</td>
            <td style="text-align: center;">53.35</td>
        </tr>
        <tr>
            <td>T-STBP</td>
            <td style="text-align: center;">137.8</td>
            <td style="text-align: center;">120.58</td>
            <td style="text-align: center;">53.00</td>
            <td style="text-align: center;">71.03</td>
            <td style="text-align: center;">23.00</td>
            <td style="text-align: center;">51.50</td>
        </tr>
        <tr>
            <td>E-prop</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">125.54</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">52.88</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">50.85</td>
        </tr>
        <tr>
            <td>OTTT</td>
            <td style="text-align: center;">141.77</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">44.61</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">17.20</td>
            <td style="text-align: center;">-</td>
        </tr>
        <tr>
            <td>SLTT</td>
            <td style="text-align: center;">149.86</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">40.53</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">15.50</td>
            <td style="text-align: center;">-</td>
        </tr>
    </tbody>
</table>


- Results for different **neuron models** across benchmark suites.

<table border="1" align="center">
    <thead>
        <tr>
            <th>Network</th>
            <th colspan="2" style="text-align: center;">PTB ($T=70$)</th>
            <th colspan="2" style="text-align: center;">PS-MNIST ($T=784$)</th>
            <th colspan="2" style="text-align: center;">Binary Adding ($T=100$)</th>
        </tr>
        <tr>
            <th>Metric</th>
            <th style="text-align: center;">FF</th>
            <th style="text-align: center;">Rec.</th>
            <th style="text-align: center;">FF</th>
            <th style="text-align: center;">Rec.</th>
            <th style="text-align: center;">FF</th>
            <th style="text-align: center;">Rec.</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>#Params.</strong></td>
            <td style="text-align: center;">~5M</td>
            <td style="text-align: center;">~6M</td>
            <td style="text-align: center;">~90K</td>
            <td style="text-align: center;">~160K</td>
            <td style="text-align: center;">~20K</td>
            <td style="text-align: center;">~40K</td>
        </tr>
        <tr>
            <td>LIF</td>
            <td style="text-align: center;">129.96</td>
            <td style="text-align: center;">111.96</td>
            <td style="text-align: center;">57.45</td>
            <td style="text-align: center;">72.97</td>
            <td style="text-align: center;">29.60</td>
            <td style="text-align: center;">53.35</td>
        </tr>
        <tr>
            <td>PLIF</td>
            <td style="text-align: center;">123.76</td>
            <td style="text-align: center;">105.64</td>
            <td style="text-align: center;">55.86</td>
            <td style="text-align: center;">77.32</td>
            <td style="text-align: center;">29.40</td>
            <td style="text-align: center;">53.25</td>
        </tr>
        <tr>
            <td>ALIF</td>
            <td style="text-align: center;">113.67</td>
            <td style="text-align: center;">102.25</td>
            <td style="text-align: center;">73.90</td>
            <td style="text-align: center;">85.78</td>
            <td style="text-align: center;">40.30</td>
            <td style="text-align: center;">68.00</td>
        </tr>
        <tr>
            <td>adLIF</td>
            <td style="text-align: center;">118.52</td>
            <td style="text-align: center;"><strong>97.22</strong></td>
            <td style="text-align: center;">85.93</td>
            <td style="text-align: center;">89.53</td>
            <td style="text-align: center;">42.00</td>
            <td style="text-align: center;">99.05</td>
        </tr>
        <tr>
            <td>GLIF</td>
            <td style="text-align: center;">111.58</td>
            <td style="text-align: center;">103.07</td>
            <td style="text-align: center;">95.42</td>
            <td style="text-align: center;">95.04</td>
            <td style="text-align: center;">90.15</td>
            <td style="text-align: center;">63.60</td>
        </tr>
        <tr>
            <td>LTC</td>
            <td style="text-align: center;"><strong>104.10</strong></td>
            <td style="text-align: center;">99.09</td>
            <td style="text-align: center;">86.33</td>
            <td style="text-align: center;">90.94</td>
            <td style="text-align: center;"><strong>100.00</strong></td>
            <td style="text-align: center;"><strong>100.00</strong></td>
        </tr>
        <tr>
            <td>SPSN</td>
            <td style="text-align: center;">120.43</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">83.88</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">45.70</td>
            <td style="text-align: center;">-</td>
        </tr>
        <tr>
            <td>TCLIF</td>
            <td style="text-align: center;">286.71</td>
            <td style="text-align: center;">255.67</td>
            <td style="text-align: center;">86.81</td>
            <td style="text-align: center;">92.08</td>
            <td style="text-align: center;">19.10</td>
            <td style="text-align: center;">19.90</td>
        </tr>
        <tr>
            <td>LM-H</td>
            <td style="text-align: center;">122.69</td>
            <td style="text-align: center;">102.05</td>
            <td style="text-align: center;">77.70</td>
            <td style="text-align: center;">83.14</td>
            <td style="text-align: center;">99.25</td>
            <td style="text-align: center;">96.10</td>
        </tr>
        <tr>
            <td>CLIF</td>
            <td style="text-align: center;">128.28</td>
            <td style="text-align: center;">108.21</td>
            <td style="text-align: center;">43.90</td>
            <td style="text-align: center;">70.44</td>
            <td style="text-align: center;">19.10</td>
            <td style="text-align: center;">64.30</td>
        </tr>
        <tr>
            <td>DH-LIF</td>
            <td style="text-align: center;">115.61</td>
            <td style="text-align: center;">100.55</td>
            <td style="text-align: center;">79.12</td>
            <td style="text-align: center;">91.07</td>
            <td style="text-align: center;">98.85</td>
            <td style="text-align: center;">99.35</td>
        </tr>
        <tr>
            <td>CELIF</td>
            <td style="text-align: center;">112.35</td>
            <td style="text-align: center;">106.52</td>
            <td style="text-align: center;"><strong>97.76</strong></td>
            <td style="text-align: center;"><strong>97.66</strong></td>
            <td style="text-align: center;">48.40</td>
            <td style="text-align: center;"><strong>100.00</strong></td>
        </tr>
        <tr>
            <td>PMSN</td>
            <td style="text-align: center;">113.24</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;">96.28</td>
            <td style="text-align: center;">-</td>
            <td style="text-align: center;"><strong>100.00</strong></td>
            <td style="text-align: center;">-</td>
        </tr>
    </tbody>
</table>


- Results for different **neural architectures** across benchmark suites

<table border="1" style="width: 100%; border-collapse: collapse;" align="center">
    <thead>
        <tr>
            <th>Dataset</th>
            <th style="text-align: center;">PTB ($T=70$)</th>
            <th style="text-align: center;">PS-MNIST ($T=784$)</th>
            <th colspan="2" style="text-align: center;">Binary Adding</th>
        </tr>
        <tr>
            <th>Metric</th>
            <th style="text-align: center;">PPL $\downarrow$</th>
            <th style="text-align: center;">Acc. $\uparrow$</th>
            <th style="text-align: center;">$T$ $\uparrow$</th>
            <th style="text-align: center;">Acc. $\uparrow$</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <th><strong>#Params.</strong></th>
            <td>~5M</td>
            <td>~90K</td>
            <td colspan="2">~40K</td>
        </tr>
        <tr>
            <td>LIF</td>
            <td style="text-align: center;">129.96</td>
            <td style="text-align: center;">57.45</td>
            <td style="text-align: center;">100</td>
            <td style="text-align: center;">34.15</td>
        </tr>
        <tr>
            <td>LIF w/ DCLS-Delays</td>
            <td style="text-align: center;">89.87</td>
            <td style="text-align: center;">68.98</td>
            <td style="text-align: center;">100</td>
            <td style="text-align: center;">51.85</td>
        </tr>
        <tr>
            <td>TCN</td>
            <td style="text-align: center;">102.20</td>
            <td style="text-align: center;">95.10</td>
            <td style="text-align: center;">1200</td>
            <td style="text-align: center;">69.95</td>
        </tr>
        <tr>
            <td>SpikingTCN</td>
            <td style="text-align: center;">114.46</td>
            <td style="text-align: center;">93.76</td>
            <td style="text-align: center;">1200</td>
            <td style="text-align: center;">61.95</td>
        </tr>
        <tr>
            <td>LSTM</td>
            <td style="text-align: center;">88.08</td>
            <td style="text-align: center;">92.41</td>
            <td style="text-align: center;">2400</td>
            <td style="text-align: center;">100</td>
        </tr>
        <tr>
            <td>Gated Spiking Neuron</td>
            <td style="text-align: center;">99.98</td>
            <td style="text-align: center;">80.13</td>
            <td style="text-align: center;">1200</td>
            <td style="text-align: center;">29.85</td>
        </tr>
        <tr>
            <td>Transformer</td>
            <td style="text-align: center;">112.43</td>
            <td style="text-align: center;">97.64</td>
            <td style="text-align: center;">2400</td>
            <td style="text-align: center;">100</td>
        </tr>
        <tr>
            <td>Spike-Driven Transformer ($T_\text{in}=4$)</td>
            <td style="text-align: center;">152.41</td>
            <td style="text-align: center;">96.21</td>
            <td style="text-align: center;">2400</td>
            <td style="text-align: center;">98.15</td>
        </tr>
        <tr>
            <td>Spike-Driven Transformer ($T_\text{in}=1$)</td>
            <td style="text-align: center;">327.82</td>
            <td style="text-align: center;">95.01</td>
            <td style="text-align: center;">2400</td>
            <td style="text-align: center;">88.05</td>
        </tr>
    </tbody>
</table>


### Steps to Reproduce Results

#### Dependencies
```shell
# Environment dependencies
torch, torchvision, torchaudio

# Configuration management
toml

# Data processing
h5py, tqdm
```

#### Experiments
Each experiment in the paper has a corresponding `toml` configuration in a folder `src/benchmark/experiments/`. We also provide scripts for all experiments as follows:
- `scripts/run_01_STP_on_benchmarks.sh`
- `scripts/run_02_training_algo_on_benchmarks.sh`
- `scripts/run_03_surro_grad_on_benchmarks.sh`
- `scripts/run_04_normalization_on_benchmarks.sh`
- `scripts/run_05_spiking_neuron_on_benchmarks.sh`
- `scripts/run_06_neuron_arch_on_benchmarks.sh`

Here is an example to reproduce the experiments of spiking neuron models by executing the file `scripts/run_05_spiking_neuron_on_benchmarks.sh`,  which contains the following commands:

```shell
# PennTreebank
python runner.py --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LIF_feedforward --data_root <path_to_dataset> --device 0
python runner.py --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PTB_LIF_recurrent --data_root <path_to_dataset> --device 0

# PS-MNIST
python runner.py --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LIF_feedforward --data_root <path_to_dataset> --device 0
python runner.py --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item PSMNIST_LIF_recurrent --data_root <path_to_dataset> --device 0

# Binary Adding
python runner.py --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LIF_feedforward --data_root <path_to_dataset> --device 0
python runner.py --experiment_name 05_spiking_neuron_on_benchmarks --experiment_item BinaryAdding_LIF_recurrent --data_root <path_to_dataset> --device 0
```


## Cite & Contact

Please cite it as follows if you have adopted or contributed to this work in your research:

```latex
@article{

}
```

Please file a report on our GitHub Issues page or contact us at `chenxiang.ma@connect.polyu.hk` if you encounter any problems or have suggestions.
