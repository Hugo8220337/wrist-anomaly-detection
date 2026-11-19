import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

def _subject_counts(oof_df):
    """
    Counts TP, FP, FN, TP per subject
    returns: DataFrame with columns ['sub', 'tp', 'fp', 'fn', 'tn']
    """
    y, p = oof_df['true_is_outlier'], oof_df['pred_is_outlier']
    d = pd.DataFrame({'sub': oof_df['sub'],
                      'tp': ((p == 1) & (y == 1)), 'fp': ((p == 1) & (y == 0)),
                      'fn': ((p == 0) & (y == 1)), 'tn': ((p == 0) & (y == 0))})
    return d.groupby('sub')[['tp', 'fp', 'fn', 'tn']].sum().sort_index()

def _metrics_from_counts(c):
    """
    Computes precision, recall, f2 and mcc from counts of TP, FP, FN
    c: array of shape (n_subjects, 4) with columns [tp, fp, fn, tn]
    returns: dict with keys 'precision', 'recall', 'f2', 'mcc' and values arrays of shape (n_subjects,)
    """
    tp, fp, fn, tn = (c[..., i].astype(float) for i in range(4))
    with np.errstate(divide='ignore', invalid='ignore'):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = tp / (tp + fn)
        f2 = np.where(4 * prec + rec > 0, 5 * prec * rec / (4 * prec + rec), 0.0)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    valid = (tp + fn) > 0                         # sem positivos -> indefinido
    return {k: np.where(valid, v, np.nan) for k, v in
            {'precision': prec, 'recall': rec, 'f2': f2, 'mcc': mcc}.items()}

def _subject_bootstrap_ci(oof_df, n_boot=2000, seed=42):
    """
    Computes the bootstrap confidence interval for subject-level metrics.
    oof_df: DataFrame with columns ['sub', 'true_is_outlier', 'pred_is_outlier']
    n_boot: number of bootstrap iterations
    seed: random seed for reproducibility
    returns: dict with keys 'precision', 'recall', 'f2', 'mcc' and values tuples of (point estimate, lower bound, upper bound)
    """
    counts = _subject_counts(oof_df).values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    boot = _metrics_from_counts(counts[idx].sum(axis=1))
    point = _metrics_from_counts(counts.sum(axis=0))
    out = {}
    for m, b in boot.items():
        b = b[~np.isnan(b)]
        out[m] = (float(point[m]), np.percentile(b, 2.5), np.percentile(b, 97.5)) if len(b) else (np.nan,) * 3
    return out

def paired_subject_bootstrap(oof_a, oof_b, metric='mcc', n_boot=2000, seed=42):
    """
    Paired subject-level bootstrap (A - B), same resampled subjects for both models.
    returns: observed difference, bootstrap differences, p-value,
             observed metric of A, observed metric of B, number of common subjects.
    """
    ca, cb = _subject_counts(oof_a), _subject_counts(oof_b)
    common = ca.index.intersection(cb.index)
    ca, cb = ca.loc[common].values, cb.loc[common].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common), size=(n_boot, len(common)))
    d = (_metrics_from_counts(ca[idx].sum(axis=1))[metric]
         - _metrics_from_counts(cb[idx].sum(axis=1))[metric])
    d = d[~np.isnan(d)]

    pa = float(_metrics_from_counts(ca.sum(axis=0))[metric])
    pb = float(_metrics_from_counts(cb.sum(axis=0))[metric])
    if len(d) == 0:
        return np.nan, d, np.nan, pa, pb, len(common)

    # two-sided; ties (d == 0) count for both sides; resolution floor = 1 / n_boot
    p = max(1.0 / n_boot, min(1.0, 2 * min((d >= 0).mean(), (d <= 0).mean())))
    return pa - pb, d, p, pa, pb, len(common)

def _subject_bootstrap_auprc(oof_df, n_boot=1000, seed=42):
    """
    Bootstrap por sujeito do AUPRC. Só produz resultado quando a coluna
    'anomaly_score' está completa (detetor forneceu score_blind).
    Descarta iterações de bootstrap sem nenhum positivo (AUPRC indefinida).
    """
    if 'anomaly_score' not in oof_df.columns or oof_df['anomaly_score'].isna().any():
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    subjects = oof_df['sub'].unique()
    groups = {s: (g['true_is_outlier'].values, g['anomaly_score'].values)
              for s, g in oof_df.groupby('sub')}

    boot_scores = []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        y = np.concatenate([groups[s][0] for s in sampled])
        sc = np.concatenate([groups[s][1] for s in sampled])
        if y.sum() == 0:
            continue
        boot_scores.append(average_precision_score(y, sc))

    if not boot_scores:
        return np.nan, np.nan, np.nan

    point = average_precision_score(oof_df['true_is_outlier'].values, oof_df['anomaly_score'].values)
    return point, np.percentile(boot_scores, 2.5), np.percentile(boot_scores, 97.5)


def compute_bootstrap_ci_all_levels(loso_results, n_boot=2000, compute_scores=True, seed=42):
    results = {}
    # Compute bootstrap confidence intervals for each level (movement and trial)
    for level, key in [('movement', 'OOF_Movement'), ('trial', 'OOF_Trial')]:
        oof = loso_results.get(key)
        if oof is None:
            continue
        if oof['true_is_outlier'].sum() == 0:
            # No positives in this level, metrics are undefined
            for m in ('precision', 'recall', 'f2', 'mcc', 'auprc'):
                results[f'{level}_{m}'] = (np.nan,) * 3
            continue

        # Compute bootstrap confidence intervals for subject-level metrics
        for m, v in _subject_bootstrap_ci(oof, n_boot, seed).items():
            results[f'{level}_{m}'] = v
        results[f'{level}_auprc'] = (_subject_bootstrap_auprc(oof, n_boot, seed)
                                     if compute_scores else (np.nan,) * 3)
    return results