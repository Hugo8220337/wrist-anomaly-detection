import numpy as np
from scipy.stats import median_abs_deviation
from utils.loso_models.anomaly_detector import AnomalyDetector

class ZScoreDetector(AnomalyDetector):
    def __init__(self, metric_type='euclidean', z_threshold=3.0, **kwargs):
        super().__init__(metric_type=metric_type, **kwargs)
        self.z_threshold = z_threshold
        self.train_medians = None
        self.train_mads = None

    def _compute_raw_scores(self, X_to_evaluate):
        """
        Continuous anomaly score, BEFORE thresholding. Higher = more
        anomalous, matching the pipeline convention. Used both by
        _predict (which then thresholds it) and by score_blind.
        """
        if self.metric_type == 'precomputed':
            # DTW: |modified z-score| of the mean distance to training series
            eval_dist = np.mean(X_to_evaluate, axis=1)
            z_scores = 0.6745 * (eval_dist - self.train_medians) / self.train_mads
            return np.abs(z_scores)
        else:
            # Features: mean |modified z-score| across all kinematic features
            z_scores = 0.6745 * (X_to_evaluate - self.train_medians) / self.train_mads
            return np.mean(np.abs(z_scores), axis=1)
        
    def _fit_stats(self, X_train):
        """Calculates the median and MAD only on the training data to avoid data leakage."""
        if self.metric_type == 'precomputed':
            # DTW: Mean distance to the other training series
            train_dist = np.mean(X_train, axis=1)
            self.train_medians = np.median(train_dist)
            self.train_mads = median_abs_deviation(train_dist)
        else:
            # Features: Statistics for each column/variable
            self.train_medians = np.median(X_train, axis=0)
            self.train_mads = median_abs_deviation(X_train, axis=0)
 
            # Prevent division by zero by replacing 0 with a very small number
            if isinstance(self.train_mads, np.ndarray):
                self.train_mads[self.train_mads == 0] = 1e-6
            elif self.train_mads == 0:
                self.train_mads = 1e-6

    def _predict(self, X_to_evaluate):
        """
        Apply the training statistics to evaluate new data.
        returns: -1 for outliers and 1 for inliers
        """
        raw_scores = self._compute_raw_scores(X_to_evaluate)
        return np.where(raw_scores > self.z_threshold, -1, 1)

    def fit_predict(self, X_train):
        self._fit_stats(X_train)
        return self._predict(X_train)

    def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        self._fit_stats(X_train)
        # Em 'precomputed', X_test já vem como a distância do teste para o treino (graças ao _split_data)
        return self._predict(X_test)

    def score_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """
        Returns the continuous mean absolute modified Z-score.
        Higher value = more anomalous.
        """
        self._fit_stats(X_train)
        return self._compute_raw_scores(X_test)