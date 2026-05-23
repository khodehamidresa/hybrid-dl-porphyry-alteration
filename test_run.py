"""
Quick test script for the Hybrid CNN-GNN-Attention model.
Generates synthetic data and verifies the model builds and runs.
"""
import numpy as np
import torch
import sys

print("=" * 50)
print("Testing Hybrid CNN-GNN-Attention model environment")
print("=" * 50)

print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CPU only: {not torch.cuda.is_available()}")

# Import model components
try:
    from model import HybridModel, build_graph
    print("Model imported successfully.")
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)

# Create synthetic data
np.random.seed(42)
n_samples, input_dim, num_classes = 100, 26, 5
feats = np.random.randn(n_samples, input_dim).astype(np.float32)
coords = np.random.randn(n_samples, 3).astype(np.float32)

# Build graph
graph = build_graph(feats, coords, k=5)
print(f"Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

# Build model and forward pass
model = HybridModel(input_dim, num_classes)
model.eval()
with torch.no_grad():
    out, attn = model(graph.x, graph.edge_index)
print(f"Output shape: {out.shape}, Attention shape: {attn.shape}")

print("=" * 50)
print("All tests passed!")
print("=" * 50)
