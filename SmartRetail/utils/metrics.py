import numpy as np


def movement_level_confusion(metadata_df, pred_idx, gt_df, keys):
    """metadata_df: one row per movement, in the order of the predictions.
    pred_idx: positional indices flagged as outliers. gt_df: annotated movements."""
    gt = gt_df[keys].drop_duplicates().assign(_gt=1)
    y_true = metadata_df[keys].merge(gt, on=keys, how='left')['_gt'].fillna(0).astype(int).to_numpy()
    y_pred = np.zeros(len(metadata_df), dtype=int)
    y_pred[np.asarray(pred_idx, dtype=int)] = 1

    tp = int(((y_pred == 1) & (y_true == 1)).sum()); fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum()); tn = int(((y_pred == 0) & (y_true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    print(f"TP={tp} FP={fp} FN={fn} TN={tn} | P={prec:.2%} R={rec:.2%} F2={f2:.2%} MCC={mcc:.2f}")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=prec, recall=rec, f2=f2, mcc=mcc)