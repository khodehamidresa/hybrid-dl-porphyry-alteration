"""
Hybrid CNN-GNN-Attention model for porphyry alteration mapping.
===============================================================================
Reproducibility code for:
"A Leakage-Aware Hybrid CNN-GNN-Attention Protocol for Hydrothermal Alteration
 Mapping and Automated Geological Logging in a Porphyry System"

WHAT THIS SCRIPT DOES (matches the manuscript's Methods section exactly)
-------------------------------------------------------------------------------
1. Loads the drillhole geochemical dataset, log10(x+1)-transforms the 25
   element columns, and holds out three spatially distal boreholes as a
   blind test set (Section 2.2).
2. Runs borehole-grouped 5-fold cross-validation (StratifiedGroupKFold),
   with per-fold RobustScaler fitting and a borehole-aware early-stopping
   split carved out of each fold's training boreholes (Section 4.2).
3. Builds every graph in a leakage-safe way (Section 4.1): the k-NN graph
   used for training is built ONLY from that fold's main-training nodes.
   Early-stopping, validation, and (at the end) blind-test nodes are never
   connected to each other -- each such node is attached to the fixed
   training graph purely through outgoing k-NN edges from its nearest
   training nodes, so no information flows back into the graph.
4. Trains three architectures for the feature-ablation comparison reported
   in Table S3: the full hybrid model, a CNN-only variant, and a GNN-only
   variant.
5. Runs the borehole-dominance diagnostic reported in Section 4.1 (mean
   fraction of a sample's 15 nearest neighbors that share its borehole).
6. Computes permutation importance and mean attention weights and reports
   their Spearman rank correlation, exactly as quoted in Section 4.2
   ("Spearman rho = 0.226, p = 0.278, n = 25").
7. Trains the final model on all 85 non-blind boreholes and evaluates it
   on the three blind boreholes.

DEFAULT BEHAVIOUR VS. THE PAPER
-------------------------------------------------------------------------------
The paper's reported final model is UNWEIGHTED (Section 5.1: "The final
model (unweighted log+Z)..."). This script is therefore unweighted by
default. Two things are opt-in and do not change that default:

  --weighted            Reproduces the weighted variant from Section 5.2
                         using the exact manual weights the paper reports
                         (ARG=4.0, PRP=5.0, SER=2.5, POT=2.0, PHY=1.0).
                         These are specific to this study's five classes.

  --suggest-weights      Prints inverse-frequency class weights computed
                         from whatever dataset you point the script at,
                         and exits without training. This is a convenience
                         for people adapting the script to their own,
                         differently-balanced dataset -- it is NOT the
                         source of the paper's manual weights above, and
                         it never runs unless you ask for it.

Class names/count are read from the data (sorted, not hard-coded), so the
script does not assume exactly five classes if you point it at a different
dataset. The reported numbers in the paper were obtained with this
dataset's five classes (ARG, PHY, POT, PRP, SER).

A NOTE ON THE GCN "BATCH SIZE"
-------------------------------------------------------------------------------
Section 4.2 reports a batch size of 64 for the hybrid model. Full-graph
GCNConv layers require a single forward pass over the (small, ~6-7k node)
training subgraph per step to produce every node's embedding, so this
script does that full-graph forward pass once per epoch and then computes
the cross-entropy loss over randomly shuffled batches of 64 node indices
from that pass, backpropagating once per batch. This reproduces the
batch-size hyperparameter's effect on the optimizer while keeping the
transductive graph convolution intact; it does not use graph sub-sampling
(e.g. neighbor loaders), which the paper does not describe.

Usage
-------------------------------------------------------------------------------
    python model.py                        # unweighted, full pipeline (default)
    python model.py --weighted             # reproduces Section 5.2's weighted variant
    python model.py --no-elevation         # log-only scenario (Supplementary Table S2)
    python model.py --skip-ablation        # skip the CNN-only/GNN-only comparison
    python model.py --suggest-weights      # print suggested weights for your own data, then exit

Requirements: torch, torch-geometric, scikit-learn, scipy, pandas, numpy, openpyxl
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

warnings.filterwarnings("ignore")

# ============================== Configuration ================================
SEED = 42
INPUT_FILE = "start_cleaned.xlsx"
OUTPUT_DIR = "outputs"
BLIND_BHIDS = ["SER_11", "SER_68", "SER_76"]

ELEMENT_COLS = [
    "Ag", "Al", "As", "Ca", "Cd", "Co", "Cr", "Cu", "Fe",
    "La", "Li", "Mg", "Mn", "Mo", "Ni", "P", "Pb", "S", "Sb",
    "Sc", "Th", "V", "Y", "Yb", "Zn",
]
TARGET_COL = "ALTERATION_TYPE"
GROUP_COL = "BHID"
COORD_COLS = ["X_loc", "Y_loc", "Z_loc"]
ELEVATION_COL = "Z_loc"

K_NEIGHBORS = 15
N_CV_FOLDS = 5
BATCH_SIZE = 64
MAX_EPOCHS = 300
EARLY_STOP_PATIENCE = 20
LEARNING_RATE = 1e-3
DROPOUT = 0.3
ES_FRACTION = 0.10  # fraction of each fold's training boreholes held out for early stopping

# Manual class weights exactly as reported in the manuscript (Section 4.2).
# These are specific to this study's five classes and are only ever applied
# when --weighted is passed; they never change the default (unweighted) run.
PAPER_CLASS_WEIGHTS = {"ARG": 4.0, "PRP": 5.0, "SER": 2.5, "POT": 2.0, "PHY": 1.0}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================== Model definitions =============================
class HybridModel(nn.Module):
    """Three parallel branches -- CNN, GNN, attention -- concatenated into a
    shared classifier (Section 4.1)."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gnn1 = GCNConv(input_dim, 128)
        self.gnn2 = GCNConv(128, 128)
        self.attn_fc = nn.Linear(input_dim, input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + input_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, edge_index):
        x_cnn = self.cnn(x.unsqueeze(1)).squeeze(-1)
        x_gnn = F.relu(self.gnn1(x, edge_index))
        x_gnn = F.relu(self.gnn2(x_gnn, edge_index))
        attn = torch.softmax(self.attn_fc(x), dim=1)
        x_attn = x * attn
        combined = torch.cat([x_cnn, x_gnn, x_attn], dim=1)
        return self.classifier(combined), attn


class CNNOnlyModel(nn.Module):
    """CNN branch only, for the feature-ablation comparison (Table S3)."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(256, num_classes)
        )

    def forward(self, x, edge_index=None):
        x_cnn = self.cnn(x.unsqueeze(1)).squeeze(-1)
        return self.classifier(x_cnn), None


class GNNOnlyModel(nn.Module):
    """GNN branch only, for the feature-ablation comparison (Table S3)."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.gnn1 = GCNConv(input_dim, 128)
        self.gnn2 = GCNConv(128, 128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(256, num_classes)
        )

    def forward(self, x, edge_index):
        x_gnn = F.relu(self.gnn1(x, edge_index))
        x_gnn = F.relu(self.gnn2(x_gnn, edge_index))
        return self.classifier(x_gnn), None


# ============================== Leakage-safe graphs ============================
def build_knn_graph(coords, k=K_NEIGHBORS):
    """k-NN graph built exclusively from `coords` (used only for a fixed
    training graph -- Section 4.1)."""
    k_eff = min(k, len(coords) - 1) if len(coords) > 1 else 0
    if k_eff <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    src, dst = [], []
    for i, neigh in enumerate(idx):
        for j in neigh:
            if j != i:
                src.append(j)
                dst.append(i)
    return torch.tensor([src, dst], dtype=torch.long)


def attach_outward(train_coords, holdout_coords, k=K_NEIGHBORS, train_offset=0, holdout_offset=0):
    """One-directional attachment edges: each held-out point receives edges
    FROM its k nearest TRAINING points. No edges are created among held-out
    points, and none flow from held-out points back into the training graph."""
    k_eff = min(k, len(train_coords))
    nbrs = NearestNeighbors(n_neighbors=k_eff).fit(train_coords)
    _, idx = nbrs.kneighbors(holdout_coords)
    src, dst = [], []
    for local_h, neigh in enumerate(idx):
        h_node = holdout_offset + local_h
        for t_local in neigh:
            src.append(train_offset + t_local)
            dst.append(h_node)
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def make_training_graph(train_x, train_coords, k=K_NEIGHBORS):
    """The fixed training graph for one fold/final model: k-NN built only
    from the training nodes themselves."""
    edge_index = build_knn_graph(train_coords, k)
    x = torch.tensor(train_x, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def make_attached_graph(train_x, train_coords, holdout_x, holdout_coords, k=K_NEIGHBORS):
    """Training graph plus one holdout set (early-stopping, validation, or
    blind) attached only through outgoing edges from the training nodes.
    Returns the combined Data object and the number of training nodes,
    so callers can slice out the holdout rows of the model output."""
    n_train = len(train_x)
    train_edges = build_knn_graph(train_coords, k)
    attach_edges = attach_outward(train_coords, holdout_coords, k, train_offset=0, holdout_offset=n_train)
    edge_index = torch.cat([train_edges, attach_edges], dim=1)
    x = torch.tensor(np.vstack([train_x, holdout_x]), dtype=torch.float)
    return Data(x=x, edge_index=edge_index), n_train


# ============================== Training / evaluation ==========================
def train_model(model, train_graph, y_train, es_x, es_coords, train_coords, y_es,
                 class_weights, patience=EARLY_STOP_PATIENCE, max_epochs=MAX_EPOCHS,
                 batch_size=BATCH_SIZE, seed=SEED):
    """Train with early stopping. The ES set is attached to the training
    graph through one-directional outward edges only (never mixed into the
    training graph itself). Loss is backpropagated in shuffled mini-batches
    of node indices over full-graph forward passes (see module docstring).

    Returns (model_with_best_weights, best_epoch), where best_epoch is the
    1-based epoch count at which the best early-stopping loss was recorded.
    This epoch count is used by the final-model stage to refit a fresh model
    on every non-blind borehole (see train_fixed_epochs), so the model
    reported as final is genuinely trained on all 85 boreholes rather than
    just the ~90% subset used to pick the stopping point."""
    train_graph = train_graph.to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    es_graph, n_train = make_attached_graph(
        train_graph.x.cpu().numpy(), train_coords, es_x, es_coords
    )
    es_graph = es_graph.to(DEVICE)
    y_es_t = torch.tensor(y_es, dtype=torch.long, device=DEVICE)

    rng = np.random.RandomState(seed)
    best_es_loss = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    n = len(y_train)

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            optimizer.zero_grad()
            out, _ = model(train_graph.x, train_graph.edge_index)
            loss = criterion(out[batch_idx], y_train_t[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            es_out, _ = model(es_graph.x, es_graph.edge_index)
            es_loss = criterion(es_out[n_train:], y_es_t)

        if es_loss.item() < best_es_loss:
            best_es_loss = es_loss.item()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def train_fixed_epochs(model, graph, y, class_weights, n_epochs,
                        batch_size=BATCH_SIZE, seed=SEED):
    """Train for exactly n_epochs with no early-stopping check and no
    held-out subset -- used only to refit the final model on the complete
    85-borehole development graph once the epoch budget has already been
    selected via a borehole-held-out split (see train_model)."""
    graph = graph.to(DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    rng = np.random.RandomState(seed)
    n = len(y)

    for _epoch in range(max(int(n_epochs), 1)):
        model.train()
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            optimizer.zero_grad()
            out, _ = model(graph.x, graph.edge_index)
            loss = criterion(out[batch_idx], y_t[batch_idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict_holdout(model, train_x, train_coords, holdout_x, holdout_coords, k=K_NEIGHBORS):
    """Predict on a held-out set (validation or blind) attached outward-only
    to the fixed training graph, never to itself."""
    graph, n_train = make_attached_graph(train_x, train_coords, holdout_x, holdout_coords, k)
    graph = graph.to(DEVICE)
    model.eval()
    out, attn = model(graph.x, graph.edge_index)
    probs = F.softmax(out, dim=1).cpu().numpy()[n_train:]
    preds = out.argmax(1).cpu().numpy()[n_train:]
    mean_attn = attn.mean(dim=0).cpu().numpy() if attn is not None else None
    return preds, probs, mean_attn


def compute_permutation_importance(model, train_x, train_coords, y, k=K_NEIGHBORS):
    """Permutation importance (Eq. 4), evaluated on the training graph
    itself (no held-out attachment needed since we are probing the model
    the same way it was trained)."""
    model.eval()
    graph = make_training_graph(train_x, train_coords, k).to(DEVICE)
    with torch.no_grad():
        base_out, base_attn = model(graph.x, graph.edge_index)
        base_preds = base_out.argmax(1).cpu().numpy()
    base_acc = accuracy_score(y, base_preds)
    mean_attn = base_attn.mean(dim=0).cpu().numpy() if base_attn is not None else None

    importances = []
    for i in range(train_x.shape[1]):
        x_perm = train_x.copy()
        np.random.shuffle(x_perm[:, i])
        perm_graph = make_training_graph(x_perm, train_coords, k).to(DEVICE)
        with torch.no_grad():
            perm_out, _ = model(perm_graph.x, perm_graph.edge_index)
            perm_preds = perm_out.argmax(1).cpu().numpy()
        importances.append(base_acc - accuracy_score(y, perm_preds))
    return np.array(importances), mean_attn


# ============================== Diagnostics =====================================
def borehole_dominance_diagnostic(coords, bhids, k=K_NEIGHBORS):
    """Section 4.1's diagnostic: for each sample, what fraction of its k
    nearest neighbors (by X, Y, Z) belong to the same borehole?"""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    bhids = np.asarray(bhids)
    same_counts = np.zeros(len(coords), dtype=int)
    for i, neigh in enumerate(idx):
        neigh = [j for j in neigh if j != i][:k]
        same_counts[i] = np.sum(bhids[neigh] == bhids[i])
    mean_same = same_counts.mean()
    pct_all_same = 100.0 * np.mean(same_counts == k)
    return mean_same, 100.0 * mean_same / k, pct_all_same


def suggest_inverse_frequency_weights(y, class_names):
    """Print-only helper: inverse-frequency class weights for whatever
    dataset is loaded. Independent of PAPER_CLASS_WEIGHTS above and never
    used for training unless the user copies the numbers themselves."""
    counts = np.bincount(y, minlength=len(class_names))
    total = counts.sum()
    print("\nSuggested inverse-frequency class weights for this dataset")
    print("(informational only -- does not affect training unless you use it yourself):")
    for name, count in zip(class_names, counts):
        weight = total / (len(class_names) * max(count, 1))
        pct = 100.0 * count / total if total else 0.0
        print(f"  {name}: n={count} ({pct:.1f}%) -> suggested weight {weight:.3f}")


# ============================== Data loading ====================================
def load_data(input_file, use_elevation=True):
    df = pd.read_excel(input_file).dropna(
        subset=ELEMENT_COLS + [TARGET_COL, GROUP_COL] + COORD_COLS
    )
    class_names = sorted(df[TARGET_COL].dropna().unique().tolist())
    df[TARGET_COL] = pd.Categorical(df[TARGET_COL], categories=class_names, ordered=False)
    y_all = df[TARGET_COL].cat.codes.values

    for col in ELEMENT_COLS:
        df[col] = np.log10(df[col].clip(lower=0) + 1)

    feature_cols = ELEMENT_COLS + ([ELEVATION_COL] if use_elevation else [])
    blind_mask = df[GROUP_COL].isin(BLIND_BHIDS)
    return df, y_all, feature_cols, class_names, blind_mask


def make_borehole_aware_split(groups, y, es_fraction=ES_FRACTION, seed=SEED):
    """Split a training fold's boreholes (not individual samples) into a
    main-training set and an early-stopping set, so no borehole appears on
    both sides (Section 4.1: 'a borehole-aware early-stopping set')."""
    gss = GroupShuffleSplit(n_splits=1, test_size=es_fraction, random_state=seed)
    idx_main, idx_es = next(gss.split(np.zeros(len(y)), y, groups))
    return idx_main, idx_es


# ============================== Main pipeline ===================================
def run_cv_and_final(args):
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df, y_all, feature_cols, class_names, blind_mask = load_data(
        INPUT_FILE, use_elevation=not args.no_elevation
    )
    num_classes = len(class_names)
    print(f"Classes found in data: {class_names}")

    df_blind = df[blind_mask].copy()
    df_work = df[~blind_mask].copy()
    y_work = y_all[~blind_mask.values]
    y_blind = y_all[blind_mask.values]

    if args.suggest_weights:
        suggest_inverse_frequency_weights(y_work, class_names)
        return

    X_work = df_work[feature_cols].values
    groups = df_work[GROUP_COL].values
    coords_work = df_work[COORD_COLS].values
    X_blind = df_blind[feature_cols].values
    coords_blind = df_blind[COORD_COLS].values

    class_weights = None
    if args.weighted:
        weights = [PAPER_CLASS_WEIGHTS.get(name, 1.0) for name in class_names]
        missing = [name for name in class_names if name not in PAPER_CLASS_WEIGHTS]
        if missing:
            print(
                f"Note: --weighted was requested but the manuscript's manual weights do not "
                f"cover class(es) {missing} in this dataset; those default to weight 1.0."
            )
        class_weights = torch.tensor(weights, dtype=torch.float, device=DEVICE)
        print(f"Using manual class weights (paper Section 4.2): {dict(zip(class_names, weights))}")
    else:
        print("Training unweighted (this reproduces the paper's reported final model).")

    architectures = {"Hybrid": HybridModel}
    if not args.skip_ablation:
        architectures["CNN-only"] = CNNOnlyModel
        architectures["GNN-only"] = GNNOnlyModel

    # ---- Borehole-dominance diagnostic (Section 4.1) ----
    mean_same, pct_mean, pct_all_same = borehole_dominance_diagnostic(coords_work, groups)
    print(
        f"\nBorehole-dominance diagnostic: mean {mean_same:.2f}/{K_NEIGHBORS} nearest "
        f"neighbors share a sample's own borehole ({pct_mean:.2f}%); "
        f"{pct_all_same:.2f}% of samples have all {K_NEIGHBORS} neighbors from the same borehole."
    )

    # ---- Spatial cross-validation ----
    sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    results = {name: {"acc": [], "f1": []} for name in architectures}
    all_preds = {name: np.empty(len(y_work), dtype=int) for name in architectures}

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_work, y_work, groups), 1):
        X_tr, X_val = X_work[train_idx], X_work[val_idx]
        y_tr, y_val = y_work[train_idx], y_work[val_idx]
        c_tr, c_val = coords_work[train_idx], coords_work[val_idx]
        g_tr = groups[train_idx]

        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        idx_main, idx_es = make_borehole_aware_split(g_tr, y_tr, seed=SEED + fold)

        for name, ModelClass in architectures.items():
            model = ModelClass(len(feature_cols), num_classes).to(DEVICE)
            train_graph = make_training_graph(X_tr_sc[idx_main], c_tr[idx_main])
            model, _fold_epoch = train_model(
                model, train_graph, y_tr[idx_main],
                X_tr_sc[idx_es], c_tr[idx_es], c_tr[idx_main], y_tr[idx_es],
                class_weights,
            )
            preds, _, _ = predict_holdout(
                model, X_tr_sc[idx_main], c_tr[idx_main], X_val_sc, c_val
            )
            all_preds[name][val_idx] = preds
            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            results[name]["acc"].append(acc)
            results[name]["f1"].append(f1)
            print(f"[{name}] Fold {fold}/{N_CV_FOLDS} | Acc: {acc:.4f} | Macro F1: {f1:.4f}")

    summary_rows = []
    for name in architectures:
        acc_arr, f1_arr = np.array(results[name]["acc"]), np.array(results[name]["f1"])
        print(f"\n{name}: CV Acc {acc_arr.mean():.4f} +/- {acc_arr.std():.4f} | "
              f"CV Macro F1 {f1_arr.mean():.4f} +/- {f1_arr.std():.4f}")
        summary_rows.append({
            "Model": name, "CV_Acc_mean": acc_arr.mean(), "CV_Acc_std": acc_arr.std(),
            "CV_MacroF1_mean": f1_arr.mean(), "CV_MacroF1_std": f1_arr.std(),
        })

    cm = confusion_matrix(y_work, all_preds["Hybrid"], labels=range(num_classes))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_excel(
        os.path.join(OUTPUT_DIR, "CV_Confusion_Matrix_Hybrid.xlsx")
    )

    # ---- Final model: pick an epoch budget, then retrain on ALL 85 boreholes ----
    # Stage A: a borehole-held-out split is used only to decide how many
    # epochs to train for (early stopping). Stage B then discards that split
    # and retrains a brand-new model on every non-blind sample for exactly
    # that many epochs, so the model reported as "final" has genuinely seen
    # all 85 development boreholes' labels, not just ~90% of them.
    print("\nSelecting the early-stopping epoch budget for the final model...")
    scaler_final = RobustScaler()
    X_work_sc = scaler_final.fit_transform(X_work)
    X_blind_sc = scaler_final.transform(X_blind)

    idx_main_f, idx_es_f = make_borehole_aware_split(groups, y_work, seed=SEED)
    epoch_selection_model = HybridModel(len(feature_cols), num_classes).to(DEVICE)
    epoch_selection_train_graph = make_training_graph(X_work_sc[idx_main_f], coords_work[idx_main_f])
    _, epoch_budget = train_model(
        epoch_selection_model, epoch_selection_train_graph, y_work[idx_main_f],
        X_work_sc[idx_es_f], coords_work[idx_es_f], coords_work[idx_main_f], y_work[idx_es_f],
        class_weights,
    )
    print(f"Selected epoch budget: {epoch_budget} (from a borehole-held-out split of the 85 boreholes)")

    print("Refitting the final model on all 85 development boreholes...")
    final_full_graph = make_training_graph(X_work_sc, coords_work)
    final_model = HybridModel(len(feature_cols), num_classes).to(DEVICE)
    final_model = train_fixed_epochs(
        final_model, final_full_graph, y_work, class_weights, epoch_budget,
    )

    # Blind holdout attaches to the FIXED FINAL TRAINING GRAPH built from all
    # non-blind boreholes (Section 4.1), i.e. all of X_work_sc -- which is
    # now also exactly what final_model's weights were trained on.
    blind_preds, blind_probs, _ = predict_holdout(
        final_model, X_work_sc, coords_work, X_blind_sc, coords_blind
    )
    blind_acc = accuracy_score(y_blind, blind_preds)
    present = np.unique(y_blind)
    blind_f1_present = f1_score(y_blind, blind_preds, labels=present, average="macro", zero_division=0)
    print(f"\nBlind Accuracy: {blind_acc:.4f} | Blind Macro F1 (classes present): {blind_f1_present:.4f}")

    pd.DataFrame([{
        "Blind_Accuracy": blind_acc, "Blind_MacroF1_present": blind_f1_present,
    }]).to_excel(os.path.join(OUTPUT_DIR, "Blind_Test_Results.xlsx"), index=False)
    pd.DataFrame(summary_rows).to_excel(
        os.path.join(OUTPUT_DIR, "CV_Architecture_Comparison.xlsx"), index=False
    )

    # ---- Permutation importance vs. attention weights (Section 4.2) ----
    # The manuscript's reported statistic excludes elevation, comparing only the
    # 25 element columns ("Elevation (Z_loc) is excluded, leaving n = 25 elements").
    print("\nComputing permutation importance and attention weights (Section 4.2)...")
    importances, mean_attn = compute_permutation_importance(final_model, X_work_sc, coords_work, y_work)
    if mean_attn is not None:
        element_idx = [i for i, c in enumerate(feature_cols) if c in ELEMENT_COLS]
        element_names = [feature_cols[i] for i in element_idx]
        rho, pval = spearmanr(mean_attn[element_idx], importances[element_idx])
        print(f"Spearman rank correlation (attention vs. permutation importance, "
              f"elevation excluded): rho = {rho:.3f}, p = {pval:.3f}, n = {len(element_idx)}")
        pd.DataFrame({
            "Feature": element_names,
            "Attention_weight": mean_attn[element_idx],
            "Permutation_importance": importances[element_idx],
        }).to_excel(os.path.join(OUTPUT_DIR, "Attention_vs_PermutationImportance.xlsx"), index=False)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weighted", action="store_true",
                    help="Apply the manuscript's manual class weights (Section 4.2 weighted variant).")
    p.add_argument("--no-elevation", action="store_true",
                    help="Exclude elevation (log-only scenario, Supplementary Table S2).")
    p.add_argument("--skip-ablation", action="store_true",
                    help="Skip the CNN-only/GNN-only comparison and only run the Hybrid model.")
    p.add_argument("--suggest-weights", action="store_true",
                    help="Print inverse-frequency class weights for the loaded dataset and exit "
                         "(does not train anything).")
    return p.parse_args()


if __name__ == "__main__":
    run_cv_and_final(parse_args())
