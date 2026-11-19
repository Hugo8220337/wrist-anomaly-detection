import numpy as np
import pandas as pd

def make_corr_filter(feature_names, threshold=0.85, prefer_drop=()):
    """Pearson filter fitted on the training movements of each fold."""
    names = list(feature_names)
    base = lambda c: c[:-2] if c.endswith(('_L', '_R')) else c # remove left/right suffixes

    def prepare(data_matrix, train_idx):
        data_matrix = np.asarray(data_matrix)
        corr = pd.DataFrame(data_matrix[train_idx], columns=names).corr().abs()
        drop = set()
        # drop features that are correlated with each other, preferring to drop those in prefer_drop
        for i, c1 in enumerate(names):
            # pairs of the same feature in both wrists are never compared
            for c2 in names[:i]:
                b1, b2 = base(c1), base(c2)
                if b1 == b2 or b1 in drop or b2 in drop or not corr.loc[c1, c2] > threshold:
                    continue
                drop.add(b1 if b1 in prefer_drop or b2 not in prefer_drop else b2)
        keep = [i for i, c in enumerate(names) if base(c) not in drop]
        return data_matrix[:, keep], sorted(drop)
    return prepare