# multimodal-seq2seq-brain-encoding
This repository contains code used in the paper “A Multimodal Sequence-to-Sequence Model for Cross-Subject Prediction of Brain Responses to Naturalistic Stimuli" by Qianyi He, Monica D. Rosenberg and Yuan Chang Leong.


## Installation

Create a conda environment and install the required packages:

```bash
conda create -n multimodal-seq2seq python=3.10
conda activate multimodal-seq2seq
pip install -r requirements.txt
```

## Data

We use the Algonauts Project 2025 release of the Courtois NeuroMod dataset. Download the challenge data from the official Algonauts Project 2025 website and place the `fmri/` and `stimuli/` directories in the repository root.

The analyses use `sub-01`, `sub-02`, `sub-03`, and `sub-05`, with fMRI responses represented in the Schaefer 1000-parcel atlas.

The main data split is:

- Friends Seasons 1–5: training
- Friends Season 6 Episodes 1–12: validation
- Friends Season 6 Episodes 13–24: in-distribution test
- Bourne, Figures, Life, and Wolf: out-of-distribution test

## Stimulus features

The models use visual, audio, language, and vision-language features extracted with VideoMAE, HuBERT, Qwen, and BridgeTower.

Feature extraction scripts are in `features/`:

```bash
python features/extract_videomae_features.py
python features/extract_hubert_features.py
python features/extract_qwen_features.py
python features/extract_bridgetower_features.py
```

The feature files used by the models are expected under:

```text
stimuli/stimulus_features/raw/
├── visual_avg/
├── audio_layer3/
├── audio_layer9/
├── language_context/
├── language_current/
└── language_visual/
```

Together, these features form a 6,656-dimensional stimulus representation at each TR.

## Building the datasets

`build_dataset.py` combines the stimulus features and parcel-wise fMRI responses into the HDF5 files used for training and evaluation.

The training split is built by default:

```bash
python build_dataset.py
```

To build the validation and test splits:

```bash
python -c "import build_dataset; build_dataset.main('val')"
python -c "import build_dataset; build_dataset.main('test')"
```

This creates:

```text
full_episode_train_data.h5
full_episode_train_meta.json
full_episode_val_data.h5
full_episode_val_meta.json
full_episode_test_data.h5
full_episode_test_meta.json
```

The test file contains both the held-out Friends episodes and the four movies.

## Training

The scripts for the main model comparisons are:

```text
train_ridge_regression.py
train_seq2one_model.py
train_seq2seq_individual_model.py
train_seq2seq_shared_model.py
train_seq2seq_hybrid_model.py
```

Run the corresponding script to train each model. For example:

```bash
python train_seq2seq_hybrid_model.py
```

The individual models are trained separately for each participant. The shared model uses a single set of parameters across participants. The hybrid model is trained jointly across participants with subject-specific embeddings and output projections.

The temporal settings selected on the validation set are:

- ridge: 15-TR stimulus context, predicting TR 15;
- seq2one: 30-TR stimulus context, predicting TR 25;
- seq2seq: 45-TR stimulus context, predicting TRs 11–40 with a 5-TR stride.

The model definitions are in `models/`. Trained checkpoints used for the reported results are provided in `checkpoints/`.

## Evaluation

`evaluation.py` generates predictions and evaluates the Transformer models on the held-out Friends episodes and movies.

Choose the model at the top of the script:

```python
MODEL_TYPE = "seq2seq_hybrid"  # seq2one | seq2seq_individual | seq2seq_hybrid | seq2seq_shared
```

Then run:

```bash
python evaluation.py
```

For Friends, valid time points are concatenated across held-out episodes before computing parcel-wise Pearson correlations. For Movies, correlations are computed separately for Bourne, Figures, Life, and Wolf. All correlations are Fisher transformed, with movie correlations averaged across films in Fisher-z space. The first and last 5 TRs of each segment are excluded from evaluation.

Per-subject parcel results are saved as Fisher z values:

```text
sub-01_parcel_z.npy
sub-02_parcel_z.npy
sub-03_parcel_z.npy
sub-05_parcel_z.npy
```

For example, evaluating the hybrid model creates:

```text
results_seq2seq_hybrid_prediction/
results_seq2seq_hybrid_friends/
results_seq2seq_hybrid_movies/
```

## Visualization

Whole-brain prediction maps can be generated with `visualization.py`.

Choose the model and test set:

```python
MODEL_TYPE = "seq2seq_hybrid"  # seq2one | seq2seq_individual | seq2seq_hybrid | seq2seq_shared
MODE = "friends"               # friends | movies
```

Then run:

```bash
python visualization.py
```

For the group maps, parcel-wise correlations are averaged across participants in Fisher-z space and transformed back to Pearson \(r\) for visualization.