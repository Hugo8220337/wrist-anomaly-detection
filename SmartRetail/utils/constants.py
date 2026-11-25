from dataclasses import dataclass
from glob import glob
from pathlib import Path

@dataclass
class Configs:
    @dataclass(frozen=True)
    class Experiment:
        exp_id: str      # "E1", "E2", "E3"
        input_: str      # "R_wrist", "both_max", "both_mean"
        gt_key: str      # "R", "L", "OR"
        dtw_loader: str  # nome do loader de matrizes

    EXPERIMENTS = [
        Experiment("E1", "R_wrist",  "R",  "right_wrist"),
        Experiment("E2", "both_max", "R",  "late_fusion"),
        Experiment("E3", "both_max", "OR", "late_fusion"),
    ]

    FEATURE_COLS = ['duration', 'range_x_L', 'range_y_L', 'range_z_L', 'path_length_L', 'path_ratio_L', 'mean_vel_L', 'max_vel_L', 'time_to_peak_vel_L', 'mean_to_peak_vel_ratio_L', 'vel_skewness_L', 'n_velocity_peaks_L', 'mean_acc_L', 'max_acc_L', 'max_jerk_L', 'ldlj_L', 'sparc_L', 'total_curvature_L', 'total_torsion_L', 'range_x_R', 'range_y_R', 'range_z_R', 'path_length_R', 'path_ratio_R', 'mean_vel_R', 'max_vel_R', 'time_to_peak_vel_R', 'mean_to_peak_vel_ratio_R', 'vel_skewness_R', 'n_velocity_peaks_R', 'mean_acc_R', 'max_acc_R', 'max_jerk_R', 'ldlj_R', 'sparc_R', 'total_curvature_R', 'total_torsion_R', 'wrist_dist_mean', 'wrist_dist_min', 'wrist_dist_max', 'wrist_dist_std', 'peak_time_diff', 'vel_corr_LR']
    KEYS = ['sub', 'repID', 'movID']
    META_COLS = ['sub', 'repID', 'movID', 'movLabel']
    PREFER_DROP = ('path_length', 'vel_skewness', 'mean_acc', 'max_acc') # features that are preferred to be dropped if feature_filter is used


    DTW_RADIUS_Seconds = [0.1, 0.2, 0.3, 0.4, 0.5]
    FS = 60 # 60Hz  (ex: 0.2 * 60 = 12 samples)
    
    CONTAMINATION_GRID = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.12, 0.14, 0.16, 0.18, 0.20                                
    ]

    FEATURES_CORRELATION_THRESHOLD = 0.85 # threshold for feature correlation filtering inside each LOSO fold
    
    # Run Time Options
    SAVE_PLOTS = True
    DESKTOP_PATH = Path.home() / "Desktop"
    
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SMART_RETAIL_BASE_DIR = Path(__file__).resolve().parent.parent
    SMART_RETAIL_DTW_BASE_DIR = (SMART_RETAIL_BASE_DIR / "SmartRetail_DTW").resolve()
    

    # data dirs
    _smart_retail_data_dir = (SMART_RETAIL_BASE_DIR / "data").resolve()
    _smart_retail_processed_data_dir = (_smart_retail_data_dir / "processed").resolve()

    _smart_reatil_dtw_data_dir = (SMART_RETAIL_DTW_BASE_DIR / "data").resolve()
    _smart_retail_dtw_processed_data_dir = (_smart_reatil_dtw_data_dir / "processed").resolve()
    
    
    # cahce dirs
    _cache_dir = (SMART_RETAIL_DTW_BASE_DIR / "cache").resolve()
    _cache_metadata_dir = (_cache_dir / "metadata").resolve()
    _cache_models_dir = (_cache_dir / "models").resolve()
    _loso_results_cache_dir = (_cache_metadata_dir / "loso_results").resolve()
    _print_log_dir = (_cache_metadata_dir / "print_logs").resolve()
        
    _smart_retail_data_path = (BASE_DIR / "Data" / "SmartRetail").resolve()
    _smart_reatail_outliers_path = (_smart_retail_data_path / "outliers").resolve()


    # SMART REAIL DATA
    SMART_RETAIL_LEFT_FILES = glob(str(_smart_retail_data_path / "*_left_Tasks.csv"))
    SMART_RETAIL_RIGHT_FILES = glob(str(_smart_retail_data_path / "*_right_Tasks.csv"))
    SMART_RETAIL_FILTERED_DATA_BIMANUAL = (_smart_retail_dtw_processed_data_dir / "filtered_data_bimanual.csv").resolve()
    
    # SMART RETAIL OUTLIERS
    SMART_RETAIL_OUTLIERS_LEFT_PATH = str(_smart_reatail_outliers_path / "outliers_left.csv")
    SMART_RETAIL_OUTLIERS_RIGHT_PATH = str(_smart_reatail_outliers_path / "outliers_right.csv")


    
    @staticmethod
    def loso_results_path(model_name: str, representation: str, experiment_name: str, movement: str) -> Path:
        return (Configs._loso_results_cache_dir / f"wrist_only_loso_results_{model_name}_{representation}_{experiment_name}_{movement}.pkl").resolve()

    @staticmethod
    def loso_model_summary_path(representation: str, gt_Key: str) -> Path:
        return (Configs._smart_retail_dtw_processed_data_dir / f"loso_model_summary_{representation}_GT{gt_Key}.csv").resolve()

    @staticmethod
    def print_log_path(model_name: str, representation: str, movement: str) -> Path:
        return (Configs._print_log_dir / f"wrist_only_loso_results_{model_name}_{representation}_{movement}.log").resolve()


    @staticmethod
    def final_model_file(model_name: str, representation: str, movement: str) -> Path:
        return (Configs._cache_models_dir / f"final_model_labels_{model_name}_{representation}_{movement}.pkl").resolve()
    

    # Feature Extraction

    _wrist_only_feature_extraction_cache_dir = (_cache_metadata_dir / "wrist_only_feature_extraction").resolve()
    @staticmethod
    def wrist_only_feature_extraction_features(movement: str) -> Path:
        return (Configs._wrist_only_feature_extraction_cache_dir / f"wrist_only_feature_extraction_features_{movement}.pkl").resolve()
    
    # DTW
    DTW_RADIUS_Seconds = [0.1, 0.2, 0.3, 0.4, 0.5]
    FS = 60 # 60Hz  (ex: 0.2 * 60 = 12 samples)

    CONTAMINATION_GRID = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
        0.12, 0.14, 0.16, 0.18, 0.20                                
    ]

    ## Mean DTW and Late Fusion DTW distance matrices
    @staticmethod
    def mean_dtw_matrix_path(movement: str) -> Path:
        return (Configs._cache_metadata_dir / f"dtw_mean_distance_matrix_{movement}.pkl").resolve()
    
    @staticmethod
    def late_fusion_dtw_matrix_path(movement: str, window_radius: float) -> Path:
        return (Configs._cache_metadata_dir / f"dtw_late_fusion_distance_matrix_{movement}_{window_radius:.2f}s.pkl").resolve()

    @staticmethod
    def dtw_matrix_6d_path(movement: str) -> Path:
        return (Configs._cache_metadata_dir / f"dtw_6d_distance_matrix_{movement}.pkl").resolve()
    

    INLIERS_MEAN_DTW_DIR = SMART_RETAIL_BASE_DIR / "data" / "processed" / "inliers_mean_dtw"
    INLIERS_LATE_FUSION_DTW_DIR = SMART_RETAIL_BASE_DIR / "data" / "processed" / "inliers_late_fusion_dtw"
    @staticmethod
    def inliers_mean_dtw_path(movement: str) -> Path:
        return (Configs.INLIERS_MEAN_DTW_DIR / f"df_cleaned_inliers_{movement}_only.csv").resolve()
    @staticmethod
    def inliers_late_fusion_dtw_path(movement: str) -> Path:
        return (Configs.INLIERS_LATE_FUSION_DTW_DIR / f"df_cleaned_inliers_{movement}_only.csv").resolve()

    # -----------------------------------------------------

    # PCA
    SMART_RETAIL_PCA_DIR = (SMART_RETAIL_BASE_DIR / "SmartRetail_PCA").resolve()
    SEPARATED_INLIERS_PCA_DIR = (SMART_RETAIL_BASE_DIR / "data" / "processed" / "separated_inliers_pca").resolve()
    @staticmethod
    def movement_inliers_pca_path(movement: str) -> Path:
        return (Configs.SMART_RETAIL_PCA_DIR / "data" / "processed" / f"df_cleaned_inliers_{movement}_only.csv").resolve()


    # UMAP
    SMART_RETAIL_UMAP_DIR = (SMART_RETAIL_BASE_DIR / "SmartRetail_UMAP").resolve()
    SEPARATED_INLIERS_UMAP_DIR = (SMART_RETAIL_BASE_DIR / "data" / "processed" / "separated_inliers_umap").resolve()
    @staticmethod
    def movement_inliers_umap_path(movement: str) -> Path:
        return (Configs.SMART_RETAIL_UMAP_DIR / "data" / "processed" / f"df_cleaned_inliers_{movement}_only.csv").resolve()
    

    def ensure_required_directories() -> None:
        directories = [
            # Data dirs
            Configs._smart_retail_data_dir,
            Configs._smart_retail_processed_data_dir,

            Configs._smart_reatil_dtw_data_dir,
            Configs._smart_retail_dtw_processed_data_dir,

            # Cache dirs
            Configs._cache_dir,
            Configs._cache_metadata_dir,
            Configs._cache_models_dir,
            Configs._loso_results_cache_dir,
            Configs._print_log_dir,
            Configs._wrist_only_feature_extraction_cache_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

Configs.ensure_required_directories()

# print(f"SMART_RETAIL_OUTLIERS_LEFT_PATH: {Configs.SMART_RETAIL_OUTLIERS_LEFT_PATH}")
# print(f"SMART_RETAIL_OUTLIERS_RIGHT_PATH: {Configs.SMART_RETAIL_OUTLIERS_RIGHT_PATH}")