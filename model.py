"""
Hybrid CNN–GNN–Attention Model for Porphyry Alteration Mapping
===========================================================================
Reproducibility script for:
"Interpretable Hybrid Deep Learning Reveals Potential Hidden Pathfinders
 and Geochemical Transition Zones in a Porphyry System"

Requirements: torch, torch-geometric, scikit-learn, pandas, numpy, openpyxl
Usage: Place `start_cleaned.xlsx` in the same directory and run:
       python model.py
"""

import os
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import warnings
warnings.filterwarnings('ignore')

# ======================== Reproducibility ========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ======================== Configuration ========================
INPUT_FILE   = "start_cleaned.xlsx"
OUTPUT_DIR   = "Output_Correct_Hybrid"
BLIND_BHIDS  = ['SER_11', 'SER_68', 'SER_76']

ELEMENT_COLS = [
    'Ag','Al','As','Ca','Cd','Co','Cr','Cu','Fe',
    'La','Li','Mg','Mn','Mo','Ni','P','Pb','S','Sb',
    'Sc','Th','V','Y','Yb','Zn'
]
TARGET       = 'ALTERATION_TYPE'
CLASS_NAMES  = ['ARG', 'PHY', 'POT', 'PRP', 'SER']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================== Model Definition ========================
class HybridModel(nn.Module):
    """Hybrid CNN–GNN–Attention model for geochemical classification."""

    def __init__(self, input_dim, num_classes):
        super().__init__()
        # CNN branch – nonlinear feature combiner
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        # GNN branch – spatial context propagator
        self.gnn1 = GCNConv(input_dim, 128)
        self.gnn2 = GCNConv(128, 128)
        # Attention branch – adaptive feature weighting
        self.attn_fc = nn.Linear(input_dim, input_dim)
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(128 + 128 + input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, edge_index):
        x_cnn = self.cnn(x.unsqueeze(1)).squeeze()
        x_gnn = F.relu(self.gnn1(x, edge_index))
        x_gnn = F.relu(self.gnn2(x_gnn, edge_index))
        attn = torch.softmax(self.attn_fc(x), dim=1)
        x_attn = x * attn
        combined = torch.cat([x_cnn, x_gnn, x_attn], dim=1)
        return self.fc(combined), attn


def build_graph(features, coords, k=15):
    """Construct a k-NN graph from spatial coordinates."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    edge_index = [[i, j] for i, neigh in enumerate(indices) for j in neigh]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)


def train_model(model, train_graph, y_train, es_graph, y_es,
                optimizer, criterion, patience=20, max_epochs=300):
    """Train with early stopping and return best model."""
    best_es_loss = float('inf')
    wait = 0
    best_state = None

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        out, _ = model(train_graph.x, train_graph.edge_index)
        loss = criterion(out, torch.tensor(y_train, dtype=torch.long, device=DEVICE))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            es_out, _ = model(es_graph.x, es_graph.edge_index)
            es_loss = criterion(es_out, torch.tensor(y_es, dtype=torch.long, device=DEVICE))

        if es_loss < best_es_loss:
            best_es_loss = es_loss
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, graph):
    """Return class predictions, probabilities, and attention weights."""
    model.eval()
    out, attn = model(graph.x, graph.edge_index)
    probs = F.softmax(out, dim=1).cpu().numpy()
    preds = out.argmax(1).cpu().numpy()
    return preds, probs, attn.mean(dim=0).cpu().numpy()


def compute_permutation_importance(model, X, coords, y, feature_cols, k=15):
    """Calculate feature importance via permutation."""
    model.eval()
    base_graph = build_graph(X, coords, k).to(DEVICE)
    with torch.no_grad():
        base_out, _ = model(base_graph.x, base_graph.edge_index)
        base_preds = base_out.argmax(1).cpu().numpy()
    base_acc = accuracy_score(y, base_preds)

    importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        perm_graph = build_graph(X_perm, coords, k).to(DEVICE)
        with torch.no_grad():
            perm_out, _ = model(perm_graph.x, perm_graph.edge_index)
            perm_preds = perm_out.argmax(1).cpu().numpy()
        importances.append(base_acc - accuracy_score(y, perm_preds))
    return importances


# ======================== Data Loading ========================
print("Loading data...")
df = pd.read_excel(INPUT_FILE).dropna(subset=ELEMENT_COLS + [TARGET, 'BHID', 'Z_loc'])
df[TARGET] = pd.Categorical(df[TARGET], categories=CLASS_NAMES, ordered=False)
y_all = df[TARGET].cat.codes.values

blind_mask = df['BHID'].isin(BLIND_BHIDS)
df_blind = df[blind_mask].copy()
df_work  = df[~blind_mask].copy()

# Log-transform
for col in ELEMENT_COLS:
    df_work[col]  = np.log10(df_work[col].clip(0) + 1)
    df_blind[col] = np.log10(df_blind[col].clip(0) + 1)

feature_cols = ELEMENT_COLS + ['Z_loc']
X_work  = df_work[feature_cols].values
y_work  = y_all[~blind_mask]
groups  = df_work['BHID'].values
coords_work = df_work[['X_loc','Y_loc','Z_loc']].values

X_blind     = df_blind[feature_cols].values
y_blind     = y_all[blind_mask]
coords_blind = df_blind[['X_loc','Y_loc','Z_loc']].values
blind_info  = df_blind[['BHID','FROM','TO','Z_loc']].copy()

class_weights = torch.tensor([4.0, 1.0, 2.0, 5.0, 2.5], dtype=torch.float, device=DEVICE)

# ======================== Spatial Cross-Validation ========================
print("\n" + "=" * 50)
print("Starting 5-fold spatial cross-validation...")
print("=" * 50)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_accuracies, fold_macro_f1s = [], []
all_preds  = np.empty(len(y_work), dtype=int)
all_probs  = np.zeros((len(y_work), len(CLASS_NAMES)))
cv_attention_accum = np.zeros((5, len(feature_cols)))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_work, y_work, groups), 1):
    X_tr, X_val = X_work[train_idx], X_work[val_idx]
    y_tr, y_val = y_work[train_idx], y_work[val_idx]
    c_tr, c_val = coords_work[train_idx], coords_work[val_idx]

    # Per-fold scaling
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    # Train / early-stopping split
    idx_main, idx_es = train_test_split(
        np.arange(len(y_tr)), test_size=0.1, stratify=y_tr, random_state=fold
    )

    train_graph = build_graph(X_tr_sc[idx_main], c_tr[idx_main], k=15).to(DEVICE)
    es_graph    = build_graph(X_tr_sc[idx_es],   c_tr[idx_es],   k=15).to(DEVICE)
    val_graph   = build_graph(X_val_sc,           c_val,          k=15).to(DEVICE)

    model = HybridModel(len(feature_cols), len(CLASS_NAMES)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=0.001)
    crit  = nn.CrossEntropyLoss(weight=class_weights)

    model = train_model(model, train_graph, y_tr[idx_main],
                        es_graph, y_tr[idx_es], opt, crit)

    preds_val, probs_val, attn = predict(model, val_graph)
    all_preds[val_idx] = preds_val
    all_probs[val_idx] = probs_val
    cv_attention_accum[fold-1] = attn

    acc = accuracy_score(y_val, preds_val)
    mf1 = f1_score(y_val, preds_val, average='macro')
    fold_accuracies.append(acc)
    fold_macro_f1s.append(mf1)
    print(f"Fold {fold}/5 | Accuracy: {acc:.4f} | Macro F1: {mf1:.4f}")

print(f"\nCV Mean Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"CV Mean Macro F1: {np.mean(fold_macro_f1s):.4f} ± {np.std(fold_macro_f1s):.4f}")

# ======================== Save CV Results ========================
pd.DataFrame({
    'Fold': range(1, 6),
    'Accuracy': fold_accuracies,
    'Macro_F1': fold_macro_f1s
}).to_excel(os.path.join(OUTPUT_DIR, 'CV_Folds_Results.xlsx'), index=False)

cm = confusion_matrix(y_work, all_preds)
pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_excel(
    os.path.join(OUTPUT_DIR, 'CV_Confusion_Matrix.xlsx'))

# ======================== Final Model (85 holes) ========================
print("\n" + "=" * 50)
print("Training final model on all 85 holes...")
print("=" * 50)

scaler_final = RobustScaler()
X_work_sc = scaler_final.fit_transform(X_work)

idx_main, idx_es = train_test_split(
    np.arange(len(y_work)), test_size=0.1, stratify=y_work, random_state=SEED
)

train_graph_final = build_graph(X_work_sc[idx_main], coords_work[idx_main], k=15).to(DEVICE)
es_graph_final    = build_graph(X_work_sc[idx_es],   coords_work[idx_es],   k=15).to(DEVICE)

final_model = HybridModel(len(feature_cols), len(CLASS_NAMES)).to(DEVICE)
opt_final   = torch.optim.Adam(final_model.parameters(), lr=0.001)
crit_final  = nn.CrossEntropyLoss(weight=class_weights)

final_model = train_model(final_model, train_graph_final, y_work[idx_main],
                          es_graph_final, y_work[idx_es], opt_final, crit_final)

# ======================== Blind Test ========================
print("\n" + "=" * 50)
print("Evaluating on blind boreholes...")
print("=" * 50)

X_blind_sc = scaler_final.transform(X_blind)
blind_graph = build_graph(X_blind_sc, coords_blind, k=15).to(DEVICE)

blind_preds, blind_probs, _ = predict(final_model, blind_graph)

blind_acc = accuracy_score(y_blind, blind_preds)
present_labels = np.unique(y_blind)
blind_macro_present = f1_score(y_blind, blind_preds, labels=present_labels, average='macro')
blind_macro_5 = f1_score(y_blind, blind_preds, labels=np.arange(len(CLASS_NAMES)),
                         average='macro', zero_division=0)

print(f"Blind Accuracy:          {blind_acc:.4f}")
print(f"Blind Macro F1 (present): {blind_macro_present:.4f}")
print(f"Blind Macro F1 (all 5):   {blind_macro_5:.4f}")

pd.DataFrame({
    'Metric': ['Accuracy', 'Macro_F1_present', 'Macro_F5'],
    'Value':  [blind_acc, blind_macro_present, blind_macro_5]
}).to_excel(os.path.join(OUTPUT_DIR, 'Blind_Test_Results.xlsx'), index=False)

blind_out = blind_info.copy()
blind_out[TARGET]     = [CLASS_NAMES[i] for i in y_blind]
blind_out['Predicted'] = [CLASS_NAMES[i] for i in blind_preds]
blind_out.to_excel(os.path.join(OUTPUT_DIR, 'Blind_Test_Predictions.xlsx'), index=False)

# ======================== Output Files ========================
print("\nGenerating supplementary output files...")

error_flags = (y_work != all_preds).astype(int)
pd.DataFrame({
    'Z_loc': df_work['Z_loc'].values,
    'Error': error_flags,
    'True_Class': [CLASS_NAMES[i] for i in y_work],
    'Predicted_Class': [CLASS_NAMES[i] for i in all_preds]
}).to_excel(os.path.join(OUTPUT_DIR, 'Elevation_Error_ECDF_Data.xlsx'), index=False)

mean_attn = cv_attention_accum.mean(axis=0)
pd.DataFrame({'Element': feature_cols, 'Weight': mean_attn}).to_excel(
    os.path.join(OUTPUT_DIR, 'Attention_Weights.xlsx'), index=False)

error_bh = df_work[['X_loc','Y_loc','Z_loc','BHID']].copy()
error_bh['Error'] = error_flags
error_bh.to_excel(os.path.join(OUTPUT_DIR, 'Spatial_Error_Map.xlsx'), index=False)

importances = compute_permutation_importance(final_model, X_work_sc, coords_work,
                                            y_work, feature_cols, k=15)
feat_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
feat_imp_df.sort_values('Importance', ascending=False).to_excel(
    os.path.join(OUTPUT_DIR, 'Feature_Importance.xlsx'), index=False)

print(f"\n{'=' * 50}")
print("All outputs saved to:", OUTPUT_DIR)
print(f"{'=' * 50}")
