import numpy as np
from sklearn.svm import OneClassSVM
from utils.loso_models.anomaly_detector import AnomalyDetector


class OCSVMDetector(AnomalyDetector):
     def __init__(self, metric_type='euclidean', **kwargs):
        super().__init__(metric_type, **kwargs)

        # gamma_scale is ours, never passed to sklearn
        self.gamma_scale = kwargs.get('gamma_scale', 1.0)
        self.params.pop('gamma_scale', None)

        if self.metric_type == 'precomputed':
            # gamma is computed from the training distances at fit time
            self.params.pop('gamma', None)
            self.params['kernel'] = 'precomputed'
        else:
            self.params['kernel'] = 'rbf'

     def _fit_gamma(self, X_train):
        """gamma = scale / median(D^2), using only the training pairs (no diagonal)."""
        iu = np.triu_indices_from(X_train, k=1)
        median_sq = np.median(X_train[iu] ** 2)
        self.gamma_ = self.gamma_scale / median_sq if median_sq > 0 else self.gamma_scale

     def _rbf_kernel(self, distance_matrix):
        """K(x, y) = exp(-gamma * D(x, y)^2), with D the raw (unsquared) DTW distance."""
        return np.exp(-self.gamma_ * distance_matrix ** 2)

     def _kernels(self, X_train, X_test=None):
        if self.metric_type != 'precomputed':
            return X_train, X_test
        self._fit_gamma(X_train)
        k_train = self._rbf_kernel(X_train)
        k_test = self._rbf_kernel(X_test) if X_test is not None else None
        return k_train, k_test

     def _compute_rbf_kernel(self, distance_matrix):
        """
        Converts a Distance Matrix into an RBF Kernel Matrix.
        Formula: K(x,y) = exp(-gamma * D(x,y))
        """
        if self.metric_type == 'precomputed':
            gamma_val = self.gamma if isinstance(self.gamma, (int, float)) else 1.0
            exponent = -gamma_val * distance_matrix
            return np.exp(exponent)
        return distance_matrix
            
     def fit_predict(self, X_train):
        k_train, _ = self._kernels(X_train)
        self.model = OneClassSVM(**self.params)
        return self.model.fit_predict(k_train)

     def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        k_train, k_test = self._kernels(X_train, X_test)
        self.model = OneClassSVM(**self.params)
        self.model.fit(k_train)
        return self.model.predict(k_test)

     def score_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """Negated decision_function: higher = more anomalous."""
        k_train, k_test = self._kernels(X_train, X_test)
        model = OneClassSVM(**self.params)
        model.fit(k_train)
        return -model.decision_function(k_test)