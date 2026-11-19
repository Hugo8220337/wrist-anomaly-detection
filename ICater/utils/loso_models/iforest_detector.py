from sklearn.ensemble import IsolationForest

from utils.loso_models.anomaly_detector import AnomalyDetector


class IForestDetector(AnomalyDetector):
    def __init__(self, metric_type='euclidean', **kwargs):
        super().__init__(metric_type=metric_type, **kwargs)

    def _build_model(self):
        return IsolationForest(**self.params, n_jobs=-1)

    def fit_predict(self, X_train):
        self.model = self._build_model()

        # if the DTW matrix is passed, X_train will be the DTW matrix. Treating each distance as a feature
        return self.model.fit_predict(X_train)

    def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        self.model = self._build_model()
        self.model.fit(X_train)
        return self.model.predict(X_test)

    def score_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """
        Returns continuous anomaly scores. Scikit-learn's IsolationForest
        returns lower values for anomalies. We invert the sign so that
        higher scores = more anomalous, matching the pipeline convention.
        """
        model = self._build_model()
        model.fit(X_train)
        return -model.score_samples(X_test)