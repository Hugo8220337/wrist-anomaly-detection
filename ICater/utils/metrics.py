def show_feature_extraction_metrics(
        features_df,
        outliers_csv,
        identified_unique,
        matched_outliers,
        missing_outliers
):
    total_trials_unique = features_df.drop_duplicates(subset=['sub', 'repID'])
    total_trials = len(total_trials_unique)

    TP = len(matched_outliers)
    FN = len(missing_outliers)  # Or len(outliers_csv) - TP
    FP = len(identified_unique) - TP
    TN = total_trials - (TP + FP + FN)

    print("--- Confusion Matrix (Based on sub+repID) ---")
    print(f"True Positives (Identified and in CSV) : {TP}")
    print(f"False Positives (Identified, not in CSV): {FP}")
    print(f"False Negatives (In CSV, not identified): {FN}")
    print(f"True Negatives (Normal, stayed normal): {TN}")
    print("-" * 50)


    print("Missing outliers (In CSV, not identified):")
    print(missing_outliers[['sub', 'repID']])


    # prevent division by zero in case the algorithm detects nothing
    if (TP + FP) > 0:
        precision = (TP / (TP + FP)) * 100
        recall = (TP / len(outliers_csv)) * 100
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f2_score = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
        fdr = (FP / (TP + FP)) * 100 # False Discovery Rate (Lixo que não era lixo)
    else:
        precision = 0.0
        recall = 0.0
        fdr = 0.0

    # Recall = TP / (TP + FN) or TP / Total in CSV
    print(f"\nPrecision (False Positives): {TP}/{TP + FP} ({precision:.2f}%)")
    print(f"Recall (True Positive Rate): {TP}/{len(outliers_csv)} ({recall:.2f}%)")
    print(f"F1-Score: {f1_score:.2f}%")
    print(f"F2-Score: {f2_score:.2f}%")

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'f2_score': f2_score,
        'fdr': fdr
    }