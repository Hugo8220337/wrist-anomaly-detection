from sklearn.neighbors import LocalOutlierFactor

from utils.loso_models.anomaly_detector import AnomalyDetector


class LOFDetector(AnomalyDetector):
    def __init__(self, metric_type='euclidean', **kwargs):
        super().__init__(metric_type, **kwargs)
 
    def fit_predict(self, X_train):
        # For standard fit_predict during hyperparameter tuning
        self.model = LocalOutlierFactor(metric=self.metric_type, **self.params, n_jobs=-1)
        return self.model.fit_predict(X_train)
 
    def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        # LOF requires novelty=True to predict on unseen data
        self.model = LocalOutlierFactor(metric=self.metric_type, novelty=True, **self.params, n_jobs=-1)
        self.model.fit(X_train)
        return self.model.predict(X_test)
 
    def score_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """
        sklearn's score_samples returns LOWER (more negative) values for
        more anomalous points. We invert the sign to follow the pipeline
        convention (higher = more anomalous).
        Uses a novelty=True model trained on X_train -- it does not
        recompute the tuning fit_predict; this is an independent training
        run used only for this blind call, just like predict_blind does.
        """
        model = LocalOutlierFactor(metric=self.metric_type, novelty=True, **self.params, n_jobs=-1)
        model.fit(X_train)
        return -model.score_samples(X_test)
 
