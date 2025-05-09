<p align="center">
  <h1 align="center">Learning to be Smooth:<br>An End-to-End Differentiable Particle Smoother</h1>
  <p align="center">
    <a href="https://asyounis.github.io/">Ali&nbsp;Younis</a>
    ·
    <a href="https://ics.uci.edu/~sudderth/">Erik&nbsp;Sudderth</a>
  </p>
  <h2 align="center">NeurIPS 2024</h2>
  <h3 align="center">
    <a href="https://asyounis.github.io/assets/neurips_2024_paper/paper.pdf">Paper</a> 
    | <a href="https://asyounis.github.io/mdps/">Project Page</a>
    | <a href="https://neurips.cc/virtual/2024/poster/94821">Video</a>
  </h3>
  <div align="center"></div>
</p>
<p align="center">
    <a href="https://asyounis.github.io/mdps/"><img src="assets/images/cover.png" alt="cover_img" width="100%"></a>
    <br>
    <em>Mixture Density Particles Smoother is an end-to-end differentiable particle smoother which uses learned dynamics and measurement models (neural networks), trained end-to-end within the particle smoother algorithm. </em>
</p>

This repository hosts the source code for "Learning to be Smooth: An End-to-End Differentiable Particle Smoother" which introduces Mixture Density Particle Smoothers (MDPS).  MDPS leverages the power of deep learning within a novel end-to-end differentiable particle smoothing algorithm to yield a superior state estimation system that maintains multiple modes in uncertain settings. This repo applies the MDPS to the simple bearings only task as well as the complex task of global localization within city scale environments.



## Installation

This repo requires Python >= 3.10 and [PyTorch](https://pytorch.org/).  We have exported the python environment using "pip3 freeze" and provided it in requirements.txt.

To run the evaluation and training, install the requirements:

```bash
python -m pip install -r requirements.txt
```

## Dataset Download and Preparation

#### Bearings Only
Bearings only dataset is a synthetic dataset that is created when any of the bearings only experiments is run for the first time.  As such please only run 1 bearings only experiment at first and once the dataset is generated and saved it is safe to run multiple bearings only jobs in parallel.

#### Mapillary Geo-Location

1. Clone the [Orienternet](https://github.com/facebookresearch/OrienterNet) repository
2. Change line 137 of `Orienternet/maploc/data/mapillaty/prepare.py` from `True` to `False`
```
# Delete
do_legacy_pano_offset": True

# Replace with
do_legacy_pano_offset": False,
```
3. Use the instructions from [Orienternet](https://github.com/facebookresearch/OrienterNet) to download the Mapillary Geo-Location dataset.
4. Place the downloaded downloaded dataset in `./data/MGL/`.
5. Download and unpack dataset split files
```
./download_MGL_splits.bash

```

#### Kitti
Use the instructions from [Orienternet](https://github.com/facebookresearch/OrienterNet) to download the KITTI dataset and then place the downloaded dataset in `./data/kitti/`.


## Training And Evaluation
All experiment configs and run scripts are contained in `./experiments/`. There are sub-directories for each dataset type.  To run an experiment navigate to that experiments directory and execute the run script. Running an experiment will result in training and evaluation of the method. Enabling or disabling training or evaluation (as well as changes to the experiment settings) can be done by editing that experiments `config.yaml` file.
For example, to run the training and evaluation of MDPS for bearings only:
```
cd ./experiments/bearings_only/mdps_strat/
./run.bash
```
**Note:** you must navigate to the directory before using the run script.

#### Bearings Only

Below is a mapping between the name of each method used in the paper and the directory in which the config file and run script are placed.
| Method Name in Paper       | Directory Name (in `./experiments/bearings_only`)      |
|----------------------------|--------------------------------------------------------|
| TG-PF (Multinomial)        | mdpf_forward_truncated_gradient_resampling_multinomial |
| TG-PF (Stratified)         | mdpf_forward_truncated_gradient_resampling_stratified  |
| SR-PF (Multinomial)        | mdpf_forward_discrete_soft_resampling_multinomial      |
| SR-PF (Stratified)         | mdpf_forward_discrete_soft_resampling_stratified       |
| MDPF (Multinomial)         | mdpf_forward_multinomial                               |
| MDPF (Stratified)          | mdpf_forward_stratified                                |
| MDPF (Residual)            | mdpf_forward_multinomial                               |
| MDPF-Backward (Stratified) | mdpf_backward_stratified                               |
| FFBS (Multinomial)         | traditional_FFBS                                       |
| MDPS (Stratified)          | mdps_strat                                             |


To run training and evaluation:
```
cd ./experiments/bearings_only/<experiment_name>
./run.bash
```
#### Mapillary Geo-Location

Below is a mapping between the name of each method used in the paper and the directory in which the config file and run script are placed.
| Method Name in Paper     | Directory Name (in `./experiments/mapillary`) |
|--------------------------|-----------------------------------------------|
| MDPF                     | mdpf_forward                                  |
| MDPS                     | mdps_strat                                    |
| Dense Search             | orienternet                                   |
| Retrieval (Sliding Win.) | embedding_maps_and_images                     |
| Retrieval (PF)           | gaussian_dynamics_pf                          |

To run training and evaluation:
```
cd ./experiments/mapillary/<experiment_name>
./run.bash
```

#### KITTI
**Note: KITTI uses the output of the Mapillary training as a starting point and you must first run the Mapillary experiments**

Below is a mapping between the name of each method used in the paper and the directory in which the config file and run script are placed.
| Method Name in Paper     | Directory Name (in `./experiments/kitti`) |
|--------------------------|-----------------------------------------------|
| MDPF                     | mdpf_forward                                  |
| MDPS                     | mdps_strat                                    |
| Dense Search             | orienternet                                   |
| Retrieval (Sliding Win.) | embedding_maps_and_images                     |
| Retrieval (PF)           | gaussian_dynamics_pf                          |


To run training and evaluation:
```
cd ./experiments/kitti/<experiment_name>
./run.bash
```

## Plotting
We provide the plotting tools for plotting the experiment results after running the training and evaluation.
Plotting tools are located in `./plots`.  Simply navigate to the directory of the plot you wish to generate and use the run script:

```
# Generate Bearings only box plots
cd ./plots/bearings_only/box_plots
./run.bash

# Generate Mapillary recall curves
cd ./plots/mapillary/recall_curves
./run.bash

# Generate Mapillary example trajectory images for MDPF/MDPS
# Note you will have to change the experiment being plotted by editing line 535 in main.py within "./plots/mapillary/plot_sequences"
cd ./plots/mapillary/plot_sequences
./run.bash

# Generate KITTI recall curves
cd ./plots/kitti/recall_curves
./run.bash
```

## BibTex citation
Please consider citing our work if you use any code from this repo or ideas presented in the paper:
```
@inproceedings{younis2024mdps,
      author    = {Younis, Ali and Sudderth, Erik},
      title     = {Learning to be Smooth: An End-to-End Differentiable Particle Smoother},
      booktitle = {NeurIPS},
      year      = {2024},
}
```