import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ICater.utils.loso_models.dbscan_detector import DBSCANDetector
from utils.loso_models.anomaly_detector import AnomalyDetector

class TransductiveDBSCANDetector(DBSCANDetector):
    """
    Variante TRANSDUTIVA (o comportamento anterior): o DBSCAN é ajustado sobre
    treino + teste do fold. Só para análise de sensibilidade: o sujeito retido
    participa no clustering, por isso não é comparável com os outros detetores.
    """
 
    def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        if full_data_matrix is None or test_idx is None:
            raise ValueError("DBSCAN requires full_data_matrix and test_idx for blind prediction.")
 
        if self.metric_type != 'precomputed':
            test_idx_arr = np.asarray(test_idx)
            train_mask = np.ones(full_data_matrix.shape[0], dtype=bool)
            train_mask[test_idx_arr] = False
            train_idx_arr = np.where(train_mask)[0]
 
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(full_data_matrix[train_idx_arr])
            X_test_scaled = scaler.transform(full_data_matrix[test_idx_arr])
 
            combined = np.vstack([X_train_scaled, X_test_scaled])
            self.model = DBSCAN(metric=self.metric_type, **self.params, n_jobs=-1)
            all_labels = self.model.fit_predict(combined)
            test_labels = all_labels[len(train_idx_arr):]
        else:
            self.model = DBSCAN(metric=self.metric_type, **self.params, n_jobs=-1)
            all_labels = self.model.fit_predict(full_data_matrix)
            test_labels = all_labels[test_idx]
 
        return np.where(test_labels == -1, -1, 1)