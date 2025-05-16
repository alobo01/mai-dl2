import pickle, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pathlib

def plot_tsne(feats_pkl_path, save_to): # Renamed argument for clarity
    with open(feats_pkl_path, "rb") as f:
        d = pickle.load(f)       # dict: feats, y_true, is_known
    
    if "feats" not in d or d["feats"] is None or len(d["feats"]) == 0:
        print(f"Warning: No features found in {feats_pkl_path}. Skipping t-SNE plot.")
        return

    X = np.array(d["feats"])
    # y = np.array(d["y_true"]) # Not used for coloring in this version, but could be
    u = np.array(d["is_known"]).astype(bool)

    print(f"Running t-SNE on {X.shape[0]} samples with dimension {X.shape[1]}...")
    # Handle cases with few samples or low perplexity requirements
    perplexity_val = min(30.0, max(5.0, X.shape[0] / 4.0 -1)) # Adjust perplexity based on N
    if X.shape[0] <= perplexity_val : # Check if N > perplexity
        print(f"Warning: Number of samples ({X.shape[0]}) is too small for perplexity ({perplexity_val}). Skipping t-SNE.")
        return

    Z = TSNE(n_components=2, init="pca", perplexity=perplexity_val, learning_rate='auto', random_state=42).fit_transform(X)
    
    plt.figure(figsize=(8,8))
    # Plot known samples (where u is True)
    if np.any(u):
        plt.scatter(Z[u,0], Z[u,1], s=10, alpha=.6, label="Known")
    # Plot unknown samples (where u is False)
    if np.any(~u):
        plt.scatter(Z[~u,0], Z[~u,1], s=10, alpha=.6, label="Unknown", marker='x')
        
    plt.legend(); plt.axis("off"); plt.tight_layout()
    plt.title("t-SNE of Embeddings")
    
    save_dir = pathlib.Path(save_to).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_to)
    print(f"t-SNE plot saved to {save_to}")
    plt.close()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="Path to scores.pkl file (must contain 'feats')")
    ap.add_argument("--out", default="tsne_feats.png", help="Path to save the plot")
    args = ap.parse_args()
    
    plot_tsne(args.scores, args.out)