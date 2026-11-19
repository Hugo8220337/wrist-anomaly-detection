import time
import warnings

import optuna
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

from utils.dtw_radius import pick_radius
from utils.loso_models.anomaly_detector import AnomalyDetector

# Mute Optuna's print statements to keep my console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

MAIN_TUNNING_METRIC = 'MCC'  # the metric used to select the best hyperparameters per fold
SECONDARY_TUNNING_METRIC = 'F2-Score'  # tie-breaker 
TERCIARY_TUNNING_METRIC = 'Pct_Rejection'  # tie-breaker 

_TUNING_METRIC_ALIASES = {
    'F1-Score': 'F1',
    'F2-Score': 'F2',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'MCC': 'MCC',
    'Pct_Rejection': 'Pct_Rejection',
    'AUPRC': 'AUPRC',
    'AUROC': 'AUROC'
}

def _get_current_fold_grid(param_grid, test_subject, X_train):
    if callable(param_grid):
        current_fold_grid = param_grid(X_train) # used only on DBSCAN because of the kneedle
    elif isinstance(param_grid, dict) and test_subject in param_grid:
        current_fold_grid = param_grid[test_subject]
    else:
        current_fold_grid = param_grid
    return current_fold_grid

def _configure_fold_parameters(data_matrix, dtw_matrices, feature_filter, train_idx):
    fold_matrix, fold_radius, fold_dropped = data_matrix, None, None
    prep_start = time.perf_counter()
    if dtw_matrices is not None:
        fold_radius, fold_matrix = pick_radius(dtw_matrices, train_idx)
    if feature_filter is not None:
        fold_matrix, fold_dropped = feature_filter(data_matrix, train_idx)
    prep_time = time.perf_counter() - prep_start
    return fold_matrix,fold_radius,fold_dropped,prep_time

def _get_tuning_metric(metrics, metric_name):
    """Return a metric by its public summary name or internal fold name."""
    metric_key = _TUNING_METRIC_ALIASES.get(metric_name, metric_name)
    if metric_key not in metrics:
        available = ', '.join(sorted(metrics))
        raise ValueError(
            f"Unknown tuning metric '{metric_name}'. Available metrics: {available}."
        )
    return metrics[metric_key]


def _metric_is_minimized(metric_name):
    """Return whether lower values are better for a tuning metric."""
    return metric_name == 'Pct_Rejection'

def _sort_tuning_results(results_df):
    """Sort the tuning results by the main and secondary metrics."""
    metric_columns = [MAIN_TUNNING_METRIC, SECONDARY_TUNNING_METRIC, TERCIARY_TUNNING_METRIC]
    ascending = [_metric_is_minimized(metric) for metric in metric_columns]
    return results_df.sort_values(by=metric_columns, ascending=ascending)

def _split_data(data_matrix, metric_type, train_idx, test_idx, scale_features=False):
    if metric_type == 'precomputed':
        # For DTW
        X_train = data_matrix[np.ix_(train_idx, train_idx)]
        X_test  = data_matrix[np.ix_(test_idx, train_idx)] 
    else:
        # For Feature Extraction
        X_train = data_matrix[train_idx]
        X_test  = data_matrix[test_idx]

        if scale_features:
            # The scaler is fit ONLY on the training data from this fold.
            # The held-out subject never influences the mean/standard deviation.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)
    return X_train,X_test

def derive_trial_level_gt(known_outliers_df):
    """
    Derives ground truth at the trial level (sub, repID) from the ground truth at the movement level (sub, repID, movID).
    A trial is considered anomalous if AT LEAST ONE of its annotated movements is in the
    """
    return known_outliers_df[['sub', 'repID']].drop_duplicates().reset_index(drop=True)

def _merge_movement_labels(metadata_df, labels, known_outliers_df):
    """
    Merge predictions (-1/1) from the detector with the movement ground truth,
    row by row, WITHOUT any aggregation. metadata_df and known_outliers_df
    share the columns ['sub', 'repID', 'movID'].
    """
    temp = metadata_df.copy()
    temp['pred_is_outlier'] = (labels == -1).astype(int)
 
    gt_keys = set(zip(known_outliers_df['sub'], known_outliers_df['repID'], known_outliers_df['movID']))
    temp['true_is_outlier'] = temp.apply(
        lambda row: 1 if (row['sub'], row['repID'], row['movID']) in gt_keys else 0,
        axis=1
    )
    return temp

def aggregate_to_trial_level(movement_level_df, trial_outliers_df, score_col=None):
    """
    Aggregates movement-level predictions to trial-level predictions.
    prediction: 1 if ANY movement in the trial was marked as an outlier (max).
    score (optional): aggregated as the MAXIMUM score among the movements in the
    trial (the most anomalous movement dominates the trial's score), consistent
    with the logic of "any anomalous segment makes the trial anomalous".
    """
    agg_dict = {'pred_is_outlier': 'max'}
    if score_col is not None and score_col in movement_level_df.columns:
        agg_dict[score_col] = 'max'
 
    trial_df = movement_level_df.groupby(['sub', 'repID']).agg(agg_dict).reset_index()
 
    gt_keys = set(zip(trial_outliers_df['sub'], trial_outliers_df['repID']))
    trial_df['true_is_outlier'] = trial_df.apply(
        lambda row: 1 if (row['sub'], row['repID']) in gt_keys else 0, axis=1
    )
    return trial_df

def _calculate_metrics(labels, metadata_df, known_outliers_df, total_movements=None, scores=None):
    """
    Calculates TP, FP, FN, TN metrics derived from predictions and ground truth
    at the movement level (sub, repID, movID), and computes Precision, Recall, F1, F2, MCC, and Pct_Rejection.
 
    NOTE: when known_outliers_df contains no records for the phase
    being evaluated (case of "Place", with 0 positives), Recall and F2 remain
    undefined (NaN) instead of being forced to 0 -- the caller should handle
    this case explicitly (see run_loso_evaluation).

    """
    merged = _merge_movement_labels(metadata_df, labels, known_outliers_df)
 
    total_unique_movements = len(merged)
    if total_movements is not None and total_movements != total_unique_movements:
        raise ValueError(
            f"total_movements={total_movements} não corresponde ao número de "
            f"movimentos encontrados em metadata_df ({total_unique_movements})."
        )
 
    tp = int(((merged['pred_is_outlier'] == 1) & (merged['true_is_outlier'] == 1)).sum())
    fp = int(((merged['pred_is_outlier'] == 1) & (merged['true_is_outlier'] == 0)).sum())
    fn = int(((merged['pred_is_outlier'] == 0) & (merged['true_is_outlier'] == 1)).sum())
    tn = int(((merged['pred_is_outlier'] == 0) & (merged['true_is_outlier'] == 0)).sum())
 
    n_known_positives = int(merged['true_is_outlier'].sum())
    n_predicted_positives = int(merged['pred_is_outlier'].sum())
 
    pct_rejection = (n_predicted_positives / total_unique_movements) * 100 if total_unique_movements > 0 else 0
 
    if n_known_positives == 0:
        # No positive class in this phase/fold: positive-class metrics
        # are not computable. Return NaN instead of 0 to avoid masking
        # the absence of signal as "poor performance".
        prec, rec, f1, f2, mcc, auprc, auroc = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = tp / n_known_positives
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f2 = (5 * prec * rec) / (4 * prec + rec) if (prec + rec) > 0 else 0.0

        if scores is not None:
            auprc = average_precision_score(merged['true_is_outlier'].values, scores)
            auroc = roc_auc_score(merged['true_is_outlier'].values, scores)
        else:
            auprc, auroc = np.nan, np.nan

        try:
            mcc = matthews_corrcoef(merged['true_is_outlier'].values, merged['pred_is_outlier'].values)
        except ValueError:
            mcc = np.nan
 
    return {
        'AUPRC': auprc, 'AUROC': auroc,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': prec, 'Recall': rec, 'F1': f1, 'F2': f2, 'MCC': mcc,
        'Pct_Rejection': pct_rejection,
        'N_Known_Positives': n_known_positives,
        'N_Predicted_Positives': n_predicted_positives,
        'Scores': scores,
    }

def _evaluate_config(detector_class, params, metric_type, X_train, train_metadata, train_outliers_gt, extract_scores):
    """Trains and evaluates one config and returns (main_metric, secondary_metric), or None if invalid."""
    detector = detector_class(metric_type=metric_type, **params)
    train_labels = detector.fit_predict(X_train)

    scores = None
    if extract_scores:
        try:
            scores = detector.score_blind(X_train, X_train) if hasattr(detector, 'score_blind') else None
        except NotImplementedError:
            scores = None

    metrics = _calculate_metrics(train_labels, train_metadata, train_outliers_gt, scores=scores)
    main_metric = _get_tuning_metric(metrics, MAIN_TUNNING_METRIC)
    secondary_metric = _get_tuning_metric(metrics, SECONDARY_TUNNING_METRIC)
    tertiary_metric = _get_tuning_metric(metrics, TERCIARY_TUNNING_METRIC)

    if pd.isna(main_metric) or pd.isna(secondary_metric):
        return None
    return float(main_metric), float(secondary_metric), float(tertiary_metric)

def _do_grid_search(detector_class, param_grid, X_train, train_metadata, train_outliers_gt, metric_type, extract_scores):
    params_log = []
 
    for params in param_grid:
        try:
            main_metric, secondary_metric, terciary_metric = _evaluate_config(detector_class, params, metric_type, X_train, train_metadata, train_outliers_gt, extract_scores=extract_scores)
        except Exception as e:
            print(f"ERROR: Config failed {params}: {type(e).__name__}: {e}")
            continue
 
        params_log.append({
            'params': params,
            MAIN_TUNNING_METRIC: main_metric,
            SECONDARY_TUNNING_METRIC: secondary_metric,
            TERCIARY_TUNNING_METRIC: terciary_metric,
        })
 
    params_df = pd.DataFrame(params_log)
    if params_df.empty:
        raise RuntimeError(
            "Empty configuration grid results. Either no valid model was found, "
            "or the selected tuning metrics are undefined for this training fold."
        )
    
    best_params = _sort_tuning_results(params_df).iloc[0]['params']
    return best_params

def _do_basyan_optimization(detector_class, param_grid, X_train, train_metadata, train_outliers_gt,
                             metric_type, n_bayes_trials, extract_scores):
    def objective(trial):
        params = {}
        for key, space in param_grid.items():
            if isinstance(space, tuple):
                param_type = space[0]
                if param_type == 'int':
                    params[key] = trial.suggest_int(key, space[1], space[2])
                elif param_type == 'float':
                    params[key] = trial.suggest_float(key, space[1], space[2])
                elif param_type == 'categorical':
                    params[key] = trial.suggest_categorical(key, space[1])
            else:
                params[key] = space
 
        try:
            main_metric, secondary_metric, terciary_metric = _evaluate_config(detector_class, params, metric_type, X_train, train_metadata, train_outliers_gt, extract_scores=extract_scores)
        except Exception:
            return float('-inf')
        
        trial.set_user_attr('secondary_metric', float(secondary_metric))
        trial.set_user_attr('terciary_metric', float(terciary_metric))
        return float(main_metric)
 
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='minimize' if _metric_is_minimized(MAIN_TUNNING_METRIC) else 'maximize',
        sampler=sampler,
    )
    study.optimize(objective, n_trials=n_bayes_trials)
 
    valid_trials = [
        trial for trial in study.trials
        if trial.value is not None and np.isfinite(trial.value)
        and 'secondary_metric' in trial.user_attrs
        and 'terciary_metric' in trial.user_attrs
    ]
    if not valid_trials:
        raise RuntimeError(
            "Bayesian optimization produced no valid configuration for the selected metrics."
        )

    valid_trials.sort(
        key=lambda trial: (
            trial.value if _metric_is_minimized(MAIN_TUNNING_METRIC) else -trial.value,
            trial.user_attrs['secondary_metric'] if _metric_is_minimized(SECONDARY_TUNNING_METRIC)
            else -trial.user_attrs['secondary_metric'],
            trial.user_attrs['terciary_metric'] if _metric_is_minimized(TERCIARY_TUNNING_METRIC)
            else -trial.user_attrs['terciary_metric'],
        )
    )

    best_params = {}
    best_trial = valid_trials[0]
    for key, space in param_grid.items():
        if isinstance(space, tuple):
            best_params[key] = best_trial.params[key]
        else:
            best_params[key] = space
 
    return best_params

def _tune_hyperparameters(
    detector_class, param_grid, X_train, train_metadata, train_outliers_gt,
    metric_type, extract_scores, search_type='grid', n_bayes_trials=30
):
    start_time = time.perf_counter()
    match search_type:
        case 'grid':
            best_params = _do_grid_search(detector_class, param_grid, X_train, train_metadata, train_outliers_gt, metric_type, extract_scores=extract_scores)
        case 'bayesian':
            best_params = _do_basyan_optimization(detector_class, param_grid, X_train, train_metadata, train_outliers_gt, metric_type, n_bayes_trials, extract_scores=extract_scores)
        case _:
            raise ValueError(f"Invalid search_type '{search_type}'. Must be 'grid' or 'bayesian'.")
 
    end_time = time.perf_counter()
    return best_params, (end_time - start_time)

def _calculate_final_results_from_confusion(global_TP, global_FP, global_FN, global_TN, total_known_positives):
    if total_known_positives == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    precision = (global_TP / (global_TP + global_FP)) if (global_TP + global_FP) > 0 else 0.0
    recall = global_TP / total_known_positives
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, f2

def _get_best_hyperparameters(fold_history):
    if not fold_history:
        print("⚠️ WARNING: No valid model was found in any fold! Returning empty params.")
        return {}, 0
 
    param_counts = {}
    for fold in fold_history:
        param_tuple = tuple(sorted(fold['params'].items()))
        param_counts[param_tuple] = param_counts.get(param_tuple, 0) + 1
 
    best_overall_tuple = max(param_counts, key=param_counts.get)
    best_overall_params = dict(best_overall_tuple)
    best_overall_params_count = param_counts[best_overall_tuple]
    return best_overall_params, best_overall_params_count

def summarize_level(oof_df, level_name, total_known_positives):
    """
    Calculates final pooled metrics and returns a dictionary with the
    summary for this aggregation level (movement or trial).
    """
    y_true = oof_df['true_is_outlier'].values
    y_pred = oof_df['pred_is_outlier'].values
 
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
 
    precision, recall, f1, f2 = _calculate_final_results_from_confusion(tp, fp, fn, tn, total_known_positives)
 
    try:
        mcc = matthews_corrcoef(y_true, y_pred) if total_known_positives > 0 else np.nan
    except ValueError:
        mcc = np.nan
 
    n_rejected = int(y_pred.sum())
    n_total = len(y_pred)
    rejection_rate = (n_rejected / n_total) * 100 if n_total > 0 else 0.0
 
    summary = {
        'Level': level_name,
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'F2-Score': f2, 'MCC': mcc,
        'N_Rejected': n_rejected, 'N_Total': n_total, 'Rejection_Rate': rejection_rate,
        'N_Known_Positives': total_known_positives,
    }
 
    # Ranking metrics (only computable if there are continuous scores and >=1 positive)
    if 'anomaly_score' in oof_df.columns and oof_df['anomaly_score'].notna().all() and total_known_positives > 0:
        try:
            summary['AUPRC'] = average_precision_score(y_true, oof_df['anomaly_score'].values)
            summary['AUROC'] = roc_auc_score(y_true, oof_df['anomaly_score'].values)
        except ValueError:
            summary['AUPRC'] = np.nan
            summary['AUROC'] = np.nan
    else:
        summary['AUPRC'] = np.nan
        summary['AUROC'] = np.nan
 
    return summary

def run_loso_evaluation(
    detector_class: type[AnomalyDetector],
    param_grid: dict | list[dict],
    data_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
    known_outliers_df: pd.DataFrame,
    metric_type: str = None,
    search_type: str = 'grid',                # ['grid', 'bayesian']
    n_bayes_trials: int = 40,
    fixed_fold_params: dict = None,
    override_random_state: int = None,
    scale_features: bool = False,
    compute_trial_level: bool = True,         # deriva também a análise trial-level, sem correr LOSO outra vez
    extract_scores: bool = True,              # tenta extrair score contínuo do detector, se disponível
    print_results: bool = True,
    dtw_matrices: dict = None,                # {w_pct: NxN}; if given, the radius is chosen inside each fold
    feature_filter = None                     # function to filter features before training  
) -> dict:
    """
    Executes Leave-One-Subject-Out Cross Validation for anomaly detection.
 
    Returns a dictionary with, among others:
        - 'Movement': pooled summary at movement level 
                (Precision, Recall,F1, F2, MCC, AUPRC, AUROC, Rejection_Rate, ...)
        - 'Trial': o mesmo resumo derivado ao nível de trial (se
            compute_trial_level=True; otherwise, None)
        - 'OOF_Movement': DataFrame long-format com uma linha por movimento
            evaluated (subject, repetition, movID, true/pred/score) -- ready
            for subject-level bootstrap and disk storage.
        - 'OOF_Trial': same, aggregated to trial level (if applicable)
        - 'Fold_History': hyperparameter and F2 history per fold 
            (for reference; no longer the basis for statistical tests)
        - 'Best_Params', 'Best_Params_Count', 'Tunning_Time', 'Inference_Time'
    """

    unique_subjects = metadata_df['sub'].unique()
    total_known_positives_movement = len(known_outliers_df)

    if total_known_positives_movement == 0:
        warnings.warn(
            "known_outliers_df is empty for this phase/representation "
            "(e.g., the 'Place' phase, with no movements annotated as anomalous). "
            "Precision/Recall/F2/MCC/AUPRC/AUROC will be NaN; only "
            "Rejection_Rate and the score distribution are interpretable."
        )
 
    print(f"LOSO Cross-Validation with {len(unique_subjects)} subjects "
            f"({total_known_positives_movement} known positive movements)...")
    
    global_TP, global_FP, global_FN, global_TN = 0, 0, 0, 0
    total_tunning_time, total_inference_time = 0.0, 0.0
    fold_history = [] # For Wilcoxon and Friedman tests

    oof_rows = []  # acumula linhas long-format: sub, repID, movID, true, pred, score
    
    for test_subject in unique_subjects:
        print(f"Evaluating Subject: {test_subject}", end=' | ')
 
        train_idx = metadata_df.index[metadata_df['sub'] != test_subject].tolist()
        test_idx = metadata_df.index[metadata_df['sub'] == test_subject].tolist()
 
        if len(test_idx) == 0:
            continue
 
        train_metadata = metadata_df.iloc[train_idx].copy()
        test_metadata = metadata_df.iloc[test_idx].copy()
 
        train_outliers_gt = known_outliers_df[known_outliers_df['sub'] != test_subject]
        test_outliers_gt = known_outliers_df[known_outliers_df['sub'] == test_subject]

 
        # --- DTW radius selection using only the training subjects or feature extraction ---
        fold_matrix, fold_radius, fold_dropped, prep_time = _configure_fold_parameters(data_matrix, dtw_matrices, feature_filter, train_idx)

        X_train, X_test = _split_data(fold_matrix, metric_type, train_idx, test_idx,
                                      scale_features=scale_features)
 
        # --- Tuning (bypass if there are fixed hyperparameters per subject) ---
        if fixed_fold_params is not None:
            best_fold_params = fixed_fold_params[test_subject].copy()
            if override_random_state is not None:
                best_fold_params['random_state'] = override_random_state
            fold_tunning_time = 0.0
        else:
            # build the current fold's param_grid, which can be a callable or a dict keyed by subject
            current_fold_grid = _get_current_fold_grid(param_grid, test_subject, X_train)

            try:
                best_fold_params, fold_tunning_time = _tune_hyperparameters(
                    detector_class, current_fold_grid, X_train, train_metadata, train_outliers_gt,
                    metric_type, search_type=search_type, extract_scores=extract_scores, n_bayes_trials=n_bayes_trials
                )
            except RuntimeError as e:
                # Training fold without known positives and without a valid config:
                # mark the fold as skipped and continue to the next subject.
                print(f"⚠️ Fold '{test_subject}' skipped during tuning: {e}")
                continue
 
        total_tunning_time += fold_tunning_time + prep_time
 
        # --- Blind test on the held-out subject ---
        detector = detector_class(metric_type=metric_type, **best_fold_params)
 
        start_time = time.perf_counter()
        test_labels = detector.predict_blind(
            X_train, X_test, full_data_matrix=fold_matrix, test_idx=test_idx
        )
        end_time = time.perf_counter()
        total_inference_time += (end_time - start_time)
 
        # --- Continuous score, if the detector provides one ---
        test_scores = None
        if extract_scores and hasattr(detector, 'score_blind'):
            try:
                test_scores = detector.score_blind(
                    X_train, X_test, full_data_matrix=fold_matrix, test_idx=test_idx
                )
            except NotImplementedError:
                test_scores = None
            except Exception as e:
                print(f"⚠️ score_blind failed for {test_subject}: {type(e).__name__}: {e}")
                test_scores = None
 
        # --- Long-format logging at movement level (no aggregation) ---
        merged = _merge_movement_labels(test_metadata, test_labels, test_outliers_gt)
        merged['anomaly_score'] = test_scores if test_scores is not None else np.nan
        oof_rows.append(merged)
 
        # --- Fold metrics (for Fold_History / diagnostics, no longer for Friedman/Wilcoxon) ---
        fold_metrics = _calculate_metrics(test_labels, test_metadata, test_outliers_gt)
 
        fold_history.append({
            'subject_out': test_subject,
            'params': best_fold_params,
            'metrics': fold_metrics,
            'dtw_radius': fold_radius,
            'dropped_features': fold_dropped,
        })
 
        global_TP += fold_metrics['TP']
        global_FP += fold_metrics['FP']
        global_FN += fold_metrics['FN']
        global_TN += fold_metrics['TN']
 
    if not oof_rows:
        raise RuntimeError("No folds produced valid results -- check known_outliers_df and param_grid.")
 
    oof_movement_df = pd.concat(oof_rows, ignore_index=True)
 
    # --- Pooled summary at movement level (main) ---
    movement_summary = summarize_level(oof_movement_df, 'movement', total_known_positives_movement)
 
    # --- Pooled summary at trial level (derived, optional) ---
    trial_summary, oof_trial_df = None, None
    if compute_trial_level:
        trial_outliers_gt = derive_trial_level_gt(known_outliers_df)
        oof_trial_df = aggregate_to_trial_level(oof_movement_df, trial_outliers_gt, score_col='anomaly_score')
        trial_summary = summarize_level(oof_trial_df, 'trial', len(trial_outliers_gt))
 
    best_overall_params, best_overall_params_count = _get_best_hyperparameters(fold_history)
    radius_counts = pd.Series([f['dtw_radius'] for f in fold_history
                               if f['dtw_radius'] is not None]).value_counts().to_dict()
    most_frequent_radius = next(iter(radius_counts), None) # most frequent; None if dtw_matrices was not used
 
    if print_results:
        print("\n✅ Final Results (LOSO Cross-Validation) -- MOVEMENT level:")
        for k in ['Precision', 'Recall', 'F1-Score', 'F2-Score', 'MCC', 'AUPRC', 'AUROC', 'Rejection_Rate']:
            v = movement_summary.get(k, np.nan)
            print(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
        if trial_summary is not None:
            print("\n✅ Final Results (derived) -- TRIAL level:")
            for k in ['Precision', 'Recall', 'F1-Score', 'F2-Score', 'MCC', 'AUPRC', 'AUROC', 'Rejection_Rate']:
                v = trial_summary.get(k, np.nan)
                print(f"  {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"  {k}: {v}")
        print(f"\nTotal Tuning Time (19 Subjects): {total_tunning_time:.4f}s")
        print(f"Total Inference Time (19 Subjects): {total_inference_time:.4f}s")
        print(f"\nOverall Best Parameters (most frequent across folds): {best_overall_params}")
        print(f"Occurrences: {best_overall_params_count}")

    return {
        # --- main summary (movement-level) ---
        'Movement': movement_summary,
        'Trial': trial_summary,

        # --- backward compatibility: "loose" fields at movement level ---
        'Precision': movement_summary['Precision'],
        'Recall': movement_summary['Recall'],
        'F1-Score': movement_summary['F1-Score'],
        'F2-Score': movement_summary['F2-Score'],
        'MCC': movement_summary['MCC'],
        'AUPRC': movement_summary['AUPRC'],
        'AUROC': movement_summary['AUROC'],
        'Rejection_Rate': movement_summary['Rejection_Rate'],
        'TP': movement_summary['TP'], 'FP': movement_summary['FP'],
        'FN': movement_summary['FN'], 'TN': movement_summary['TN'],

        # --- long-format data, ready for subject-level bootstrap ---
        'OOF_Movement': oof_movement_df,
        'OOF_Trial': oof_trial_df,

        # --- compatibility with the existing compute_bootstrap_ci ---
        'Blind_Y_True': oof_movement_df['true_is_outlier'].values,
        'Blind_Y_Pred': oof_movement_df['pred_is_outlier'].values,
        'Blind_Y_Score': oof_movement_df['anomaly_score'].values,
        
        # --- diagnostics and reproducibility ---
        'Tunning_Time': total_tunning_time,
        'Inference_Time': total_inference_time,
        'Fold_History': fold_history,
        'Best_Params': best_overall_params,
        'Best_Params_Count': best_overall_params_count,
        'Most_Frequent_DTW_Radius': most_frequent_radius,
        'DTW_Radius_Counts': radius_counts,
    }