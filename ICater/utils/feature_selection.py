import numpy as np
import pandas as pd

def make_corr_filter(feature_names, threshold=0.85, prefer_drop=()):
    """Pearson filter fitted on the training movements of each fold."""
    names = list(feature_names)

    def prepare(data_matrix, train_idx):
        data_matrix = np.asarray(data_matrix)
        corr = pd.DataFrame(data_matrix[train_idx], columns=names).corr().abs()
        drop = set()
        for i, c1 in enumerate(names):
            for c2 in names[:i]:
                if c1 in drop or c2 in drop or not corr.loc[c1, c2] > threshold:
                    continue
                drop.add(c1 if c1 in prefer_drop or c2 not in prefer_drop else c2)
        keep = [i for i, c in enumerate(names) if c not in drop]
        return data_matrix[:, keep], sorted(drop)
    return prepare