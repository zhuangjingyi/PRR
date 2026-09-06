# Progressive Relighting Representation for RGB-based Low-Light Person Re-Identification

[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)

Official PyTorch implementation of the paper **Progressive Relighting Representation for RGB-based Low-Light Person Re-Identification**.

## Updates

- (xx/9/2026) Code released!
%- (xx/xx/2026) Several experiment update!

## Highlights

In this paper, we propose a **progressive relighting representation (PRR)** framework to investigate this direction. PRR establishes dense interactions between relighting and identity encoding along the network depth, allowing relighting information to participate continuously in the formation of identity representations. These interactions are implemented through a **relighting-aware query modulation (RAQM)** module embedded in each self-attention layer of the ReID encoder. At each layer, RAQM extracts relighting cues from an enhancement branch and adaptively modulates the query embeddings before attention computation. Since query embeddings directly influence which contextual information each identity token attends to, their modulation allows the attention patterns to adapt to relighting guidance during successive encoding stages, thereby steering intermediate identity representations as they evolve through the encoder. For training stability under noisy nighttime observations, we incorporate a relaxed hard-mining objective for triplet learning.

![framework](./images/framework.png)

## Usage

### Requirements

We use a single NVIDIA RTX 3090 24G GPU for training and evaluation. 

```
pytorch 1.8.2
torchvision 0.9.2
yacs
timm
scipy
scikit-image
tqdm
```

You can install all dependencies via:
```
pip install -r requirements.txt
```


### Prepare Datasets

Download **Night600** and **NightReID** and organize them in `cfg.DATASETS.ROOT_DIR` as follows:
```
|-- your dataset root dir/
| |-- Night600/
| |-- bounding_box_train
| |-- bounding_box_test
| |-- query
|
| |-- NightReID/
| |-- bounding_box_train
| |-- bounding_box_test
| |-- query
```

Set `DATASETS.ROOT_DIR` and `DATASETS.NAMES` in the config file (e.g. `configs/nightreid/PRR.yml`) accordingly. `DATASETS.EVAL_MODE` selects the NightReID evaluation protocol (`528_wD` / `528_woD` / `1000_wD` / `1000_woD`).

### Training
```
python train.py --config_file configs/nightreid/PRR.yml
```

The output directory is automatically suffixed with the key hyper-parameters (`pos{s_pos}_neg{s_neg}_Frequency{s}_Tri{lambda}_LossType_{loss_type}`) for easy bookkeeping across runs.

### Testing
