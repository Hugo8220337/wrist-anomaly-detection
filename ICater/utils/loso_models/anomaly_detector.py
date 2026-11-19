from abc import ABC, abstractmethod


class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection models.
    Defines the interface that all specific model wrappers must implement.
    """
    def __init__(self, metric_type='euclidean', **kwargs):
        self.metric_type = metric_type
        self.params = kwargs
        self.model = None

    @abstractmethod
    def fit_predict(self, X_train):
        """Fits the model on X_train and returns the outlier labels (-1 for outlier, 1 for inlier)."""
        pass

    @abstractmethod
    def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
         """
         Trains on X_train and predicts on X_test. 
         Handles edge cases where models don't have a native 'predict' method.
         """
         pass
    
    def get_params(self):
        return self.params

    def score_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """
        Returns an array of continuous scores for X_test, trained on
        X_train, where HIGHER VALUES = MORE ANOMALOUS (the convention used
        throughout the pipeline for average_precision_score/roc_auc_score).
        Not implemented by default.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement a continuous score_blind()."
        )