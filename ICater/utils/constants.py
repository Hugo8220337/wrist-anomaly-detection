from glob import glob
from pathlib import Path

class Configs:
    FEATURE_COLS = ['duration', 'range_x', 'range_y', 'range_z', 'path_length', 'path_ratio', 'mean_vel', 'max_vel', 'time_to_peak_vel', 'mean_to_peak_vel_ratio', 'vel_skewness', 'n_velocity_peaks', 'mean_acc', 'max_acc', 'max_jerk', 'ldlj', 'sparc', 'total_curvature', 'total_torsion']
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

    # Base Dirs
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    ICATER_BASE_DIR = Path(__file__).resolve().parent.parent

    ICATER_WRIST_ONLY_PIPELINE_BASE_DIR = (ICATER_BASE_DIR / "ICater_Hand_only_pipeline").resolve()
    ICATER_ALL_SENSORS_PIPELINE_BASE_DIR = (ICATER_BASE_DIR / "ICater_Shoulder_Elbow_Hand").resolve()

    # ICATER DATA
    _icater_data_path = (BASE_DIR / "Data" / "ICATER").resolve()
    _icater_outliers_path = (_icater_data_path / "outliers").resolve()
    ICATER_FILES = glob(str(_icater_data_path / "*.csv"))
    ICATER_OUTLIER_FILES = glob(str(_icater_outliers_path / "*.csv"))[0]

    # ICater Filtered Data
    _icater_processed_data_path = (ICATER_BASE_DIR / "data" / "processed").resolve()
    ICATER_WRIST_ONLY_FILTERED_DATA_DIR = (_icater_processed_data_path / "wrist_only_filtered_data.csv").resolve()
    ICATER_ALL_SENSORS_FILTERED_DATA_DIR = (_icater_processed_data_path / "all_sensors_filtered_data.csv").resolve()

    # Wrist Only Cache
    _wrist_only_cache_dir = (ICATER_WRIST_ONLY_PIPELINE_BASE_DIR / "cache").resolve()
    _wrist_only_feature_extraction_cache_dir = (_wrist_only_cache_dir / "metadata" / "wrist_only_feature_extraction").resolve()
    _wrist_only_dtw_cache_dir = (_wrist_only_cache_dir / "metadata" / "wrist_only_dtw").resolve()
    _wrist_only_loso_results_cache_dir = (_wrist_only_cache_dir / "metadata" / "loso_results").resolve()
    WRIST_ONLY_FINAL_INLIERS = (_icater_processed_data_path / "wrist_only_inliers").resolve()

    # Summary files for excel regist auxiliary 
    WRIST_ONLY_CSV_SHAPE_BASED_SUMMARY_FILE = (_icater_processed_data_path / "shape_based_model_summary.csv").resolve()
    WRIST_ONLY_FEATURE_EXTRACTION_SUMMARY_FILE = (_icater_processed_data_path / "feature_extraction_model_summary.csv").resolve()

    @staticmethod
    def wrist_only_feature_extraction_features(movement: str) -> Path:
        return (Configs._wrist_only_feature_extraction_cache_dir / f"wrist_only_feature_extraction_features_{movement}.pkl").resolve()

    @staticmethod
    def wrist_only_constrained_dtw_distance_matrix(movement: str, window_radius: float) -> Path:
        return (Configs._wrist_only_dtw_cache_dir / f"wrist_only_dtw_distance_matrix_{movement}_{window_radius:.2f}s.pkl").resolve()

    @staticmethod
    def cached_wrist_only_dtw_inliers(model_name: str, movement: str) -> Path:
        return (Configs._wrist_only_dtw_cache_dir / "inliers" / f"wrist_only_dtw_inliers_{model_name}_{movement}.pkl").resolve()

    @staticmethod
    def loso_results_path(model_name: str, representation: str, movement: str) -> Path:
        return (Configs._wrist_only_loso_results_cache_dir / f"wrist_only_loso_results_{model_name}_{representation}_{movement}.pkl").resolve()

    # All Sensors Cache
    _all_sensors_cache_dir = (ICATER_ALL_SENSORS_PIPELINE_BASE_DIR / "cache").resolve()
    _all_sensors_dtw_cache_dir = (_all_sensors_cache_dir / "metadata" / "all_sensors_dtw").resolve()
    _all_sensors_dtw_cache_inliers_dir = (_all_sensors_dtw_cache_dir / "inliers").resolve()
    ALL_SENSORS_DTW_CALCULATION_TIMES_CACHE = (_all_sensors_dtw_cache_dir / "all_sensors_dtw_calculation_times.pkl").resolve()
    ALL_SENSORS_FINAL_INLIERS = (ICATER_BASE_DIR / "data" / "processed" / "all_sensors_inliers").resolve()

    @staticmethod
    def all_sensors_dtw_distance_matrix(movement: str) -> Path:
        return (Configs._all_sensors_dtw_cache_dir / f"all_sensors_dtw_distance_matrix_{movement}.pkl").resolve()

    @staticmethod
    def cached_all_sensors_dtw_inliers(model_name: str, movement: str):
        return (Configs._all_sensors_dtw_cache_inliers_dir / f"all_sensors_dtw_inliers_{model_name}_{movement}.pkl").resolve()

    @staticmethod
    def cached_all_sensors_PCA_inliers(model_name: str, movement: str):
        return (Configs._all_sensors_dtw_cache_inliers_dir / f"all_sensors_PCA_inliers_{model_name}_{movement}.pkl").resolve()

    # ----
    _icater_final_model_dir = (_icater_processed_data_path / "final_models_wrist_only").resolve()
    @staticmethod
    def final_model_file(model_name: str, representation: str, movement: str) -> Path:
        return (Configs._icater_final_model_dir / f"final_model_labels_{model_name}_{representation}_{movement}.pkl").resolve()

    def ensure_required_directories() -> None:
        directories = [
            Configs.ICATER_BASE_DIR / "data" / "processed",
            Configs.WRIST_ONLY_FINAL_INLIERS,
            Configs.ALL_SENSORS_FINAL_INLIERS,
            Configs._icater_processed_data_path,
            Configs._wrist_only_feature_extraction_cache_dir,
            Configs._wrist_only_dtw_cache_dir,
            Configs._wrist_only_dtw_cache_dir,
            Configs._wrist_only_loso_results_cache_dir,
            Configs._all_sensors_dtw_cache_dir,
            Configs._all_sensors_dtw_cache_inliers_dir,
            Configs._icater_final_model_dir
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

Configs.ensure_required_directories()

# print(f"ICATER_WRIST_ONLY_PIPELINE_BASE_DIR: {Configs.ICATER_WRIST_ONLY_PIPELINE_BASE_DIR}")
# print(f"ICATER_BASE_DIR: {Configs.ICATER_BASE_DIR}")