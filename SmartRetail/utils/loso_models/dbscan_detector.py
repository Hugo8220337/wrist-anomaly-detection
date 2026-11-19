import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from utils.loso_models.anomaly_detector import AnomalyDetector


class DBSCANDetector(AnomalyDetector):
    def __init__(self, metric_type='euclidean', **kwargs):
        super().__init__(metric_type, **kwargs)

    def fit_predict(self, X_train):
        """
        Fit the DBSCAN model on the training data and predict labels for the training data.
        Parameters:
        - X_train: Training data (numpy array or similar).
        Returns:
        - labels: An array of labels where -1 indicates anomalies and 1 indicates normal points.
        """
        self.model = DBSCAN(metric=self.metric_type, **self.params, n_jobs=-1)
        labels = self.model.fit_predict(X_train)
        return np.where(labels == -1, -1, 1)
 
    def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        self.model = DBSCAN(metric=self.metric_type, **self.params, n_jobs=-1)
        self.model.fit(X_train)
 
        core_idx = self.model.core_sample_indices_
        if len(core_idx) == 0:
            return np.full(X_test.shape[0], -1)   # no clusters: all test points are anomalies
 
        if self.metric_type == 'precomputed':
            dist_to_core = np.asarray(X_test)[:, core_idx]          # teste x cores
        else:
            dist_to_core = pairwise_distances(
                X_test, np.asarray(X_train)[core_idx], metric=self.metric_type
            )
 
        nearest_core = dist_to_core.min(axis=1)
        return np.where(nearest_core <= self.params['eps'], 1, -1)

    def score_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """
        DBSCAN does not provide a continuous anomaly score mechanism natively in sklearn.
        Raises NotImplementedError to be handled gracefully by the pipeline.
        """
        raise NotImplementedError("DBSCAN does not support continuous anomaly scores.")