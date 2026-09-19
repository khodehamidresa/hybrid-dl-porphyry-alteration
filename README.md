# hybrid-dl-porphyry-alteration

Reproducibility code for:

**"A Leakage-Aware Hybrid CNN-GNN-Attention Protocol for Hydrothermal Alteration
Mapping and Automated Geological Logging in a Porphyry System"**

This repository contains the exact model architecture, spatial cross-validation
protocol, feature-ablation comparison, and diagnostics reported in the paper.
It does not include the geochemical dataset, which is proprietary to the
National Iranian Copper Industries Company (NICICO) — see the paper's Data
Availability statement.

## What's in this repo

- `model.py` — the full pipeline: data loading, leakage-safe graph
  construction, borehole-grouped 5-fold cross-validation, the CNN-only /
  GNN-only / Hybrid feature-ablation comparison, the borehole-dominance
  diagnostic, the permutation-importance-vs-attention comparison, and the
  final blind-test evaluation.
- `requirements.txt` — pinned minimum package versions.

## Setup

```bash
pip install -r requirements.txt
```

## Input data format

The script expects an Excel file (default name `start_cleaned.xlsx`, set by
`INPUT_FILE` at the top of `model.py`) in the working directory, with one row
per assay interval and these columns:

- `BHID` — borehole ID (string)
- `X_loc`, `Y_loc`, `Z_loc` — sample coordinates (numeric, meters)
- `ALTERATION_TYPE` — the alteration class label (string)
- The 25 element columns analyzed by ICP-MS: `Ag, Al, As, Ca, Cd, Co, Cr, Cu,
  Fe, La, Li, Mg, Mn, Mo, Ni, P, Pb, S, Sb, Sc, Th, V, Y, Yb, Zn`

Three boreholes (`SER_11`, `SER_68`, `SER_76`) are held out as the spatially
distal blind test set, matching the paper. If you run this on a different
deposit, change `BLIND_BHIDS` at the top of `model.py` to your own holdout
boreholes.

Class names and count are read from the data itself (not hard-coded), so a
dataset with a different number or naming of alteration classes will run
without modification — the manuscript's own reported numbers were obtained
with this dataset's five classes (ARG, PHY, POT, PRP, SER).

## Usage

```bash
python model.py                    # unweighted (the paper's reported final model)
python model.py --weighted         # reproduces the Section 5.2 weighted variant
python model.py --no-elevation     # log-only scenario (Supplementary Table S2)
python model.py --skip-ablation    # only train the Hybrid model, skip CNN-only/GNN-only
python model.py --suggest-weights  # print inverse-frequency weight suggestions for
                                    # YOUR dataset, then exit (does not train anything)
```

Outputs (cross-validation results, the confusion matrix, the architecture
comparison, the blind-test results, and the permutation-importance/attention
comparison) are written as `.xlsx` files to `outputs/`.

### On class weighting

The paper's final, reported model is **unweighted** — this is the default
here. `--weighted` reproduces the paper's Section 5.2 weighted variant using
the exact manual weights it reports for this dataset's five classes (ARG=4.0,
PRP=5.0, SER=2.5, POT=2.0, PHY=1.0); those numbers are specific to this
study's class balance and are not a general-purpose formula. If you're
adapting this to your own, differently-imbalanced dataset, `--suggest-weights`
prints inverse-frequency weights computed from your data instead — it's
purely informational and never affects training on its own.

### On the final model

The final model is fit in two stages: a borehole-held-out split first picks
an early-stopping epoch budget, then a fresh model is retrained from scratch
on **all 85 non-blind boreholes** for exactly that many epochs (no further
early stopping). This ensures the model evaluated on the blind boreholes has
actually seen every non-blind sample's label, not just the ~90% used to pick
the epoch count.

### On the leakage-safe graph construction

Every k-NN graph used during training, early stopping, validation, and the
final blind evaluation is built the same way the paper describes in Section
4.1: the graph is built only from the current training nodes, and every
held-out node (early-stopping, validation, or blind) is attached to that
fixed graph only through outgoing edges from its nearest training nodes —
never to other held-out nodes, and never feeding information back into the
training graph.

### On the reported "batch size 64"

`GCNConv` layers require a single forward pass over the full training
subgraph to produce every node's embedding, so this script performs that
full-graph forward pass once per epoch and then computes the cross-entropy
loss over shuffled batches of 64 node indices from that pass, backpropagating
once per batch. This reproduces the batch-size hyperparameter's effect on the
optimizer while keeping the transductive graph convolution intact.

## Code Availability statement in the manuscript

The manuscript's Code Availability section should reference only the files
actually in this repository (`model.py`, `requirements.txt`, this README) —
please keep that statement in sync with the repo contents.
