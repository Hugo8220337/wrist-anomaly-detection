import joblib
import numpy as np


def load_dtw_matrices(movement, window_percentages, path_fn):
    """Returns {w_pct: NxN matrix}, same row order as metadata_df."""
    return {w: joblib.load(str(path_fn(movement, w))) for w in window_percentages}


def pick_radius(matrices, train_idx):
    """Highest variance of the pairwise fused distances among TRAINING movements only."""
    idx = np.asarray(train_idx)
    iu = np.triu_indices(len(idx), k=1)
    scores = {w: float(np.var(m[np.ix_(idx, idx)][iu])) for w, m in matrices.items()}
    best = max(scores, key=scores.get)
    return best, matrices[best]