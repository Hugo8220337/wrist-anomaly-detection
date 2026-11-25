from dataclasses import dataclass
from glob import glob
from pathlib import Path

@dataclass
class Configs:
    BASE_DIR = Path(__file__).resolve().parent.parent

    # ICater
    _icater_data_path = (BASE_DIR / "ICATER").resolve()
    ICATER_RAW_FILES = glob(str(_icater_data_path / "*.csv"))
    ICATER_PROCESSED_FILES = glob(str(Path("/home/hugo-guimar-es/Desktop/ICater_inliers_DTW") / "*.csv"))
    ICATER_OUTLIERS_PATH = str(_icater_data_path / "outliers.csv")

    _smart_retail_data_path = (BASE_DIR / "SmartRetail").resolve()
    _smart_reatail_outliers_path = (_smart_retail_data_path / "outliers").resolve()

    # SMART REAIL DATA
    SMART_RETAIL_LEFT_FILES = glob(str(_smart_retail_data_path / "*_left_Tasks.csv"))
    SMART_RETAIL_RIGHT_FILES = glob(str(_smart_retail_data_path / "*_right_Tasks.csv"))
    
    # SMART RETAIL OUTLIERS
    SMART_RETAIL_OUTLIERS_LEFT_PATH = str(_smart_reatail_outliers_path / "outliers_left.csv")
    SMART_RETAIL_OUTLIERS_RIGHT_PATH = str(_smart_reatail_outliers_path / "outliers_right.csv")

print("ICATER RAW FILES:", Configs.ICATER_RAW_FILES)