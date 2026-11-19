import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # progress bar in notebooks
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from utils.constants import Configs


def normalize_series(
    df_input,
    movLabel_filter=None,
    useScaler=True,
    group_array=['x_clean_smooth', 'y_clean_smooth', 'z_clean_smooth'],
    sample_size=500
):
    """
    Normalize time series groups (sub, repID, movID) to zero mean and unit variance.
    params:
        - df_input: DataFrame to use (defaults to global df)
        - movLabel_filter: movLabel value to keep (e.g., 'Move','Pick','Place') or None
        - sample_size: number of rows to sample (or None to use full dataframe)
    """

    if movLabel_filter is not None:
        df_input = df_input[df_input["movLabel"] == movLabel_filter]

    if sample_size is not None:
        sample_df = df_input.sample(n=sample_size, random_state=42)
    else:
        sample_df = df_input


    groups = sample_df.groupby(['sub', 'repID', 'movID'])
    series_list = []

    for name, group in tqdm(groups, desc="Processing time series"):
        arr = group[group_array].values.astype(float)
        series_list.append(arr)

    if len(series_list) == 0:
        raise ValueError("No series found. Check df_input, sample_size and movID_filter.")

    print("Shape of first series: " + str(series_list[0].shape))

    formated_series = to_time_series_dataset(series_list)

    if useScaler:
        scaler = TimeSeriesScalerMeanVariance()
        formated_series = scaler.fit_transform(formated_series)

    print("Shape of formatted series: " + str(formated_series.shape))
    return formated_series

def calculate_trash_percentage(features_df, total_trials, anomaly_col='anomaly', outlier_value=-1):
    """
    Used in Feature Extraction context.
    Correctly calculates the percentage of unique trials discarded.
    Avoids double counting when multiple rows from the same trial are marked as outliers.
    I made this because OCSVM had 200% outlier detection rate

    params:
        - features_df: DataFrame containing the features and the 'anomaly' column
        - total_trials: total number of unique trials (sub + repID)
        - anomaly_col: name of the column indicating anomalies (default 'anomaly')
        - outlier_value: value in the anomaly column that indicates an outlier (default -
    returns: 
        - n_unique_outliers: number of unique trials marked as outliers (sub + repID)
        - trash_percentage: percentage of unique trials marked as outliers
    """
    anomalous_rows = features_df[features_df[anomaly_col] == outlier_value]
    
    # Extract the unique trials (sub + repID) affected
    unique_anomalous_trials = anomalous_rows.drop_duplicates(subset=['sub', 'repID'])
    n_unique_outliers = len(unique_anomalous_trials)
    trash_percentage = (n_unique_outliers / total_trials) * 100 if total_trials > 0 else 0
    
    return n_unique_outliers, trash_percentage

def calculate_matrix_trash_percentage(outlier_indices, metadata_df):
    """
    Used in the DTW matrix context. Calculates the percentage of unique trials discarded based on outlier indices and metadata.
    params:
        - outlier_indices: indices of the detected outliers in the data matrix
        - metadata_df: DataFrame containing the metadata with 'sub' and 'repID'
    returns:
        - n_unique: number of unique trials marked as outliers (sub + repID)
        - trash_pct: percentage of unique trials marked as outliers
    """
    unique_total_trials = len(metadata_df.drop_duplicates(subset=['sub', 'repID']))
    
    outliers_meta = metadata_df.iloc[outlier_indices]
    unique_outliers = outliers_meta.drop_duplicates(subset=['sub', 'repID'])
    
    n_unique = len(unique_outliers)
    trash_pct = (n_unique / unique_total_trials) * 100 if unique_total_trials > 0 else 0
    
    return n_unique, trash_pct


def save_loso_results(model_name, representation, target_movement, loso_results):
    """
    Extracts fold history from loso_results and saves it to a DataFrame, then saves the entire loso_results object to disk using joblib.
    args:
        model_name: name of the model (e.g., 'OCSVM', 'IsolationForest')
        representation: representation used (e.g., 'features', 'DTW_matrix')
        target_movement: the movement for which results are being saved
        loso_results: the results from the LOSO evaluation
    """
    
    # Extract fold history to a DataFrame
    fold_results = pd.DataFrame(loso_results['Fold_History'])
    fold_results['Model'] = model_name
    fold_results['Representation'] = representation
    fold_results['Phase'] = target_movement

    file_path = Configs.loso_results_path(model_name, representation, target_movement)
    joblib.dump(loso_results, file_path)

def _fmt_point_ci(point, low, upp, scale=100):
    """
    Formats a point estimate and its confidence interval for display.
    returns a string in the format: "point [low–upp]" or NaN.
    """
    if point is None or (isinstance(point, float) and np.isnan(point)):
        return "N/A"
    s = f"{point*scale:.2f}"
    if low is None or upp is None or np.isnan(low) or np.isnan(upp):
        return s
    return f"{s} [{low*scale:.2f}–{upp*scale:.2f}]"
 
def _metric_cell(level_summary, ci, level_name, summary_key, ci_key, scale=100):
    """
    Helper function to format a metric cell for display in the summary table.
    """
    _, low, upp = ci.get(f'{level_name}_{ci_key}', (None, None, None))
    return _fmt_point_ci(level_summary.get(summary_key), low, upp, scale)

def _print_level_block(level_name, phase, representation, algorithm, level_summary, ci):
    total = level_summary['N_Total']
    label = "Trials" if level_name == 'trial' else "Movements"
 
    print(f"\n✅ Experiment Summary for {algorithm} - {representation} - {phase} - {total} {label} ({level_name.upper()}-level):")
    print(f"Unique Outliers Detected: {level_summary['N_Rejected']}")
    print(f"Rejection Rate: {level_summary['Rejection_Rate']:.2f}%")
    print(f"Known Positives in Ground Truth: {level_summary['N_Known_Positives']}")
 
    print("-" * 30)
    print(f"Total True Positives (Global):  {level_summary['TP']}")
    print(f"Total False Positives (Global): {level_summary['FP']}")
    print(f"Total False Negatives (Global): {level_summary['FN']}")
    print(f"Total True Negatives (Global):  {level_summary['TN']}")
    print("-" * 30)
 
    auprc_ci = ci.get(f'{level_name}_auprc', (level_summary.get('AUPRC', np.nan), None, None))

    print(f"Blind Precision: {_metric_cell(level_summary, ci, level_name, 'Precision', 'precision')}")
    print(f"Blind Recall:    {_metric_cell(level_summary, ci, level_name, 'Recall', 'recall')}")
    print(f"Blind F1-Score:  {_metric_cell(level_summary, ci, level_name, 'F1-Score', 'f1')}")
    print(f"Blind F2-Score:  {_metric_cell(level_summary, ci, level_name, 'F2-Score', 'f2')}")
    print(f"Blind MCC:       {_metric_cell(level_summary, ci, level_name, 'MCC', 'mcc', scale=1.0)}")

    if not (isinstance(auprc_ci[0], float) and np.isnan(auprc_ci[0])):
        print(f"Blind AUPRC:  {_metric_cell(level_summary, ci, level_name, 'AUPRC', 'auprc')}")
        print(f"Blind AUROC:   {_metric_cell(level_summary, ci, level_name, 'AUROC', 'auroc')}")
    else:
        print("Blind AUPRC:     N/A (detector has no continuous score_blind, or no known positives)")
        print("Blind AUROC:     N/A")

 
def append_experiment_record(
    summary_path: str,
    phase: str,
    representation: str,
    algorithm: str,
    total_samples: int,
    loso_results: dict,
    bootstrap_results: dict = None,
):
    """
    Writes the summary CSV (one row per level: movement and trial) and
    prints a detailed summary for each level. Called ONCE per
    experiment -- no need to repeat for trial.
    """
    ci = bootstrap_results or {}
 
    records = []
    for level_name, level_summary in [('movement', loso_results['Movement']), ('trial', loso_results.get('Trial'))]:
        if level_summary is None:
            continue

        mcc = level_summary['MCC']
        if mcc is None or (isinstance(mcc, float) and np.isnan(mcc)):
            mcc = "N/A"

        prec_ci = ci.get(f'{level_name}_precision', (level_summary['Precision'], None, None))
        rec_ci = ci.get(f'{level_name}_recall', (level_summary['Recall'], None, None))
        f2_ci = ci.get(f'{level_name}_f2', (level_summary['F2-Score'], None, None))
        auprc_ci = ci.get(f'{level_name}_auprc', (level_summary.get('AUPRC', np.nan), None, None))
        mcc_ci = ci.get(f'{level_name}_mcc', (level_summary['MCC'], None, None))
 
        records.append({
            'Algorithm': algorithm, 'Representation': representation,
            'Movement Phase': phase, 'Level': level_name,
            'Num. Observations': level_summary['N_Total'],
            'N° Known Positives': level_summary['N_Known_Positives'],
            'N° Rejected': level_summary['N_Rejected'],
            'Rej. Rate (%)': round(level_summary['Rejection_Rate'], 2),
            'TP': level_summary['TP'], 'FP': level_summary['FP'],
            'FN': level_summary['FN'], 'TN': level_summary['TN'],
            'Precision (%)': _metric_cell(level_summary, ci, level_name, 'Precision', 'precision'),
            'Recall (%)':    _metric_cell(level_summary, ci, level_name, 'Recall', 'recall'),
            'F2-Score (%)':  _metric_cell(level_summary, ci, level_name, 'F2-Score', 'f2'),
            'MCC':           _metric_cell(level_summary, ci, level_name, 'MCC', 'mcc', scale=1.0),
            'AUPRC (%)':     _metric_cell(level_summary, ci, level_name, 'AUPRC', 'auprc'),
            'AUROC (%)':     _fmt_point_ci(level_summary.get('AUROC'), None, None),
            'Tunning Time (s)': round(loso_results['Tunning_Time'], 4),
            'Inference Time (s)': round(loso_results['Inference_Time'], 4),
        })
 
    df_new = pd.DataFrame(records)
 
    if os.path.exists(summary_path):
        try:
            summary_df = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            summary_df = pd.DataFrame(columns=df_new.columns)
        if not summary_df.empty:
            mask = (
                (summary_df["Algorithm"] == algorithm) &
                (summary_df["Movement Phase"] == phase) &
                (summary_df["Representation"] == representation) &
                (summary_df["Level"].isin(df_new["Level"]))
            )
            summary_df = pd.concat([summary_df[~mask], df_new], ignore_index=True)
        else:
            summary_df = df_new.copy()
    else:
        summary_df = df_new.copy()
 
    summary_df.to_csv(summary_path, index=False)
 
    # --- Print detalhado, por nível ---
    for level_name, level_summary in [('movement', loso_results['Movement']), ('trial', loso_results.get('Trial'))]:
        if level_summary is None:
            continue
        _print_level_block(level_name, phase, representation, algorithm, level_summary, ci)
 
    print(f"\nTotal Tuning Time (19 Subjects): {loso_results['Tunning_Time']:.4f}s")
    print(f"Total Inference Time (19 Subjects): {loso_results['Inference_Time']:.4f}s")
    print(f"\nOverall Best Parameters (Most frequent across folds):\n   {loso_results['Best_Params']}")
    print(f"Occurrences: {loso_results['Best_Params_Count']}")