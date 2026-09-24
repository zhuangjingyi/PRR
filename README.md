# Progressive Relighting Representation Learning for RGB-based Low-light Person Re-identification

[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red)](https://pytorch.org/)

Official PyTorch implementation of the paper **Progressive Relighting Representation Learning for RGB-based Low-light Person Re-identification**.

## Updates

- (xx/9/2026) Code released!

## Highlights

In this paper, we propose a **progressive relighting representation learning (PRRL)** framework to investigate this direction. PRRL establishes dense interactions between relighting and identity encoding along the network depth, allowing relighting information to participate continuously in the formation of identity representations. These interactions are implemented through a **relighting-aware query modulation (RAQM)** module embedded in each self-attention layer of the ReID encoder. At each layer, RAQM extracts relighting cues from an enhancement branch and adaptively modulates the query embeddings before attention computation. Since query embeddings directly influence which contextual information each identity token attends to, their modulation allows the attention patterns to adapt to relighting guidance during successive encoding stages, thereby steering intermediate identity representations as they evolve through the encoder. For training stability under noisy nighttime observations, we incorporate a relaxed hard-mining triplet (RHMT) for learning.

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

Download the datasets from their original providers:

- [NightReID](https://github.com/msm8976/NightReID)
- [Night600](https://github.com/Alexadlu/IDF)

Please follow the original dataset access requirements and usage terms.

Organize the datasets as follows:

```
your dataset root dir/
├── NightReID/
│   ├── bounding_box_train/
│   ├── query/
│   ├── query-1000/
│   ├── bounding_box_test/
│   ├── bounding_box_test-528-noDistractors/
│   ├── bounding_box_test_withDistractors/
│   └── bounding_box_test-1000-noDistractors/
└── night600/
    ├── bounding_box_train/
    ├── query_3/
    └── bounding_box_test/
```

Set `DATASETS.ROOT_DIR` and `DATASETS.NAMES` in the config file (e.g. `configs/nightreid/PRR.yml`) accordingly. `DATASETS.EVAL_MODE` selects the NightReID evaluation protocol (`528_wD` / `528_woD` / `1000_wD` / `1000_woD`).


### Training

Download the **ViT-B/16+ICS** pre-trained backbone from [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL#pre-trained-models), and set `MODEL.PRETRAIN_PATH` to the downloaded `vit_base_ics_cfs_lup.pth` file.

The default YAML configurations correspond to the full PRRL model. For ablation studies and parameter analysis, modify the relevant parameters to match the checkpoint being evaluated.

```bash
# NightReID
python train.py --config_file configs/nightreid/PRR.yml 

# Night600
python train.py --config_file configs/night600/PRR.yml
```


### Testing

After training, evaluate the saved checkpoint using the same model configuration.

```bash
# NightReID
python test.py --config_file configs/nightreid/PRR.yml 

# Night600
python test.py --config_file configs/night600/PRR.yml
```

For NightReID, change `DATASETS.EVAL_MODE` to select the evaluation protocol:

| `DATASETS.EVAL_MODE` | Evaluation protocol |
| --- | --- |
| `528_wD` | 528 identities with distractors |
| `528_woD` | 528 identities without distractors |
| `1000_wD` | 1000 identities with distractors |
| `1000_woD` | 1000 identities without distractors |


## Comparison with State-of-the-Art Methods

The following results (%) are reported in our paper.

<p align="center">
  <img src="./images/table1.png" alt="table1" width="900">
</p>
<p align="center">
  <img src="./images/table2.png" alt="table2" width="500">
</p>


## Reproducing the Paper Results

### Download Trained Checkpoints

To reproduce the reported results without retraining, download the corresponding trained checkpoints: [Baidu Netdisk](https://pan.baidu.com/s/1KYLy3Es6vteR0VAYuY8njg?pwd=646i)

The download is organized into the following experiment folders. Place them under `./logs/` and preserve their original names:

```text
logs/
├── baseline_night600/
├── baseline_nightreid/
├── relighting interval/
│   ├── night600/
│   │   ├── pos3_neg7_Frequency1_Tri1.0_LossType_prr_triplet/
│   │   ├── pos3_neg7_Frequency3_Tri1.0_LossType_prr_triplet/
│   │   └── pos3_neg7_Frequency6_Tri1.0_LossType_prr_triplet/
│   └── nightreid/
└── s_pos_s_neg/
```

The `baseline_*` folders contain baseline/ablation experiment files. The `relighting interval` and `s_pos_s_neg` folders organize the corresponding parameter experiments.

### Evaluate the Full Model

The trained models are stored at:

- **NightReID:** `./logs/s_pos_s_neg/nightreid/pos3_neg1_Frequency1_Tri1.0_LossType_prr_triplet/transformer_best_mAP.pth`
- **Night600:** `./logs/s_pos_s_neg/night600/pos3_neg7_Frequency1_Tri1.0_LossType_prr_triplet/transformer_best_mAP.pth`

```bash
# NightReID
python test.py --config_file configs/nightreid/PRR.yml \
  TEST.WEIGHT "./logs/s_pos_s_neg/nightreid/pos3_neg1_Frequency1_Tri1.0_LossType_prr_triplet/transformer_best_mAP.pth"

# Night600
python test.py --config_file configs/night600/PRR.yml \
  TEST.WEIGHT "./logs/s_pos_s_neg/night600/pos3_neg7_Frequency1_Tri1.0_LossType_prr_triplet/transformer_best_mAP.pth"
```

The default evaluation protocol is `528_wD`. To evaluate other protocols, change `DATASETS.EVAL_MODE` in `configs/nightreid/PRR.yml` to `528_woD`, `1000_wD`, or `1000_woD`.


## Ablation Studies

We evaluate the contributions of RAQM and RHMT.

<p align="center">
  <img src="./images/table3.png" alt="Ablation results" width="900">
</p>

Select the evaluation script, checkpoint directory, and mining parameters according to the table below. Parameter pairs are written as `(K_pos, K_neg)`.

| Row | Setting | Script | Checkpoint directory | NightReID | Night600 |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline&nbsp;without&nbsp;PRRL | `test_woRAQM.py` | `baseline_<dataset>/pos1_neg1` | (1, 1) | (1, 1) |
| 2 | RAQM&nbsp;only | `test.py` | `s_pos_s_neg/<dataset>/pos1_neg1_Frequency1_Tri1.0_LossType_prr_triplet` | (1, 1) | (1, 1) |
| 3 | RHMT&nbsp;only | `test_woRAQM.py` | `baseline_<dataset>/pos3_neg<K_neg>` | (3, 1) | (3, 7) |
| 4 | Full&nbsp;PRRL | `test.py` | `s_pos_s_neg/<dataset>/pos3_neg<K_neg>_Frequency1_Tri1.0_LossType_prr_triplet` | (3, 1) | (3, 7) |

All checkpoint directories are under `./logs/` and contain `transformer_best_mAP.pth`. Replace `<dataset>` with `nightreid` or `night600`, and `<K_neg>` with the corresponding value.

Use the following command, updating the variables for the desired row. This example evaluates **Row 1 on Night600**:

```bash
DATASET=night600
SCRIPT=test_woRAQM.py
POS=1
NEG=1
WEIGHT_DIR="./logs/baseline_night600/pos1_neg1"

python "$SCRIPT" --config_file "configs/${DATASET}/PRR.yml" \
  MODEL.INTERACTION_INTERVAL 1 \
  SOLVER.HARD_EXAMPLE_POS_K "$POS" \
  SOLVER.HARD_EXAMPLE_NEG_K "$NEG" \
  TEST.WEIGHT "${WEIGHT_DIR}/transformer_best_mAP.pth"
```

For NightReID, change `DATASETS.EVAL_MODE` in its YAML file to select `528_wD`, `528_woD`, `1000_wD`, or `1000_woD`.

Each row requires its corresponding trained checkpoint; changing mining parameters only during testing does not reproduce a different training setting.


## Parameter Analysis

Set the following parameters in the corresponding YAML file before evaluation:

| Parameter | Configuration key |
| --- | --- |
| Relighting interval `s` | `MODEL.INTERACTION_INTERVAL` |
| Positive mining count `K_pos` | `SOLVER.HARD_EXAMPLE_POS_K` |
| Negative mining count `K_neg` | `SOLVER.HARD_EXAMPLE_NEG_K` |

The testing script automatically loads the checkpoint using the configured parameters:

```text
{OUTPUT_DIR}/pos{K_pos}_neg{K_neg}_Frequency{s}_Tri1.0_LossType_prr_triplet/transformer_best_mAP.pth
```

### Relighting Interval

Set `MODEL.INTERACTION_INTERVAL` to `1`, `3`, or `6`.

```bash
# NightReID
python test.py --config_file configs/nightreid/PRR.yml \
  OUTPUT_DIR "./logs/relighting interval/nightreid"

# Night600
python test.py --config_file configs/night600/PRR.yml \
  OUTPUT_DIR "./logs/relighting interval/night600"
```

### Positive and Negative Mining

Keep `MODEL.INTERACTION_INTERVAL=1`.
Use the following commands for either analysis after updating the corresponding YAML parameters.

```bash
# NightReID
python test.py --config_file configs/nightreid/PRR.yml \
  OUTPUT_DIR "./logs/s_pos_s_neg/nightreid"

# Night600
python test.py --config_file configs/night600/PRR.yml \
  OUTPUT_DIR "./logs/s_pos_s_neg/night600"
```

## Acknowledgments

We thank the authors of [NightReID / EDA](https://github.com/msm8976/NightReID), [Night600 / IDF](https://github.com/Alexadlu/IDF), [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL), and [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) for sharing their code, datasets, and pretrained models.
