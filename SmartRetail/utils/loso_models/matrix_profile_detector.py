import numpy as np
import stumpy
from utils.loso_models.anomaly_detector import AnomalyDetector

# o normalize false desacelera muito o processo de treino
class MatrixProfileDetector(AnomalyDetector):
    def __init__(self, metric_type='euclidean', m=10, contamination=0.05, normalize=False, **kwargs):
        super().__init__(metric_type=metric_type, **kwargs)
        self.m = int(m)
        self.contamination = contamination
        self.threshold_ = None
        self.concat_z_train = None
        self.normalize = normalize

        # Caching last calculated test score (so STUMPY is not recomputed in score_blind
        # immediately after predict_blind in the same fold).
        self._last_scores_test = None
        self._last_test_signature = None

    def _build_concat(self, X):
        """Concatenates the Z-axis of all trajectories, separated by NaNs."""
        concat_z = []
        trial_map = []
        current_idx = 0
 
        for i in range(X.shape[0]):
            z_series = X[i, :, 2]  # index 2 is the Z axis
            z_series = z_series[~np.isnan(z_series)]
 
            start_idx = current_idx
            concat_z.extend(z_series)
            end_idx = current_idx + len(z_series)
 
            trial_map.append({'original_idx': i, 'start': start_idx, 'end': end_idx})
 
            # Spacer of NaN values to avoid that STUMPY mixes the end of a series with the beginning of another
            spacer = [np.nan] * self.m
            concat_z.extend(spacer)
            current_idx = end_idx + self.m
 
        return np.array(concat_z, dtype=float), trial_map

    def _get_scores(self, mp_distances, trial_map):
        """Extract the highest discord score (distance) per repetition."""
        scores = np.zeros(len(trial_map))
        for trial in trial_map:
            start = trial['start']
            end = trial['end']
            valid_end = end - self.m + 1
 
            if valid_end > start:
                trial_distances = mp_distances[start:valid_end]
                scores[trial['original_idx']] = np.max(trial_distances)
            else:
                scores[trial['original_idx']] = 0
        return scores

    def fit_predict(self, X_train):
        """Train and find the outliers in the training set."""
        self.concat_z_train, trial_map = self._build_concat(X_train)
 
        # Normal Matrix Profile (Série contra ela própria)
        mp = stumpy.stump(self.concat_z_train, self.m, normalize=self.normalize)
        mp_distances = np.asarray(mp[:, 0], dtype=float)
        mp_distances[np.isinf(mp_distances)] = 0.0  # STUMPY uses inf instead of NaNs
 
        scores = self._get_scores(mp_distances, trial_map)
 
        # Calculating the threshold based on contamination
        percentile_val = 100 - (self.contamination * 100)
        self.threshold_ = np.percentile(scores, percentile_val)
 
        return np.where(scores > self.threshold_, -1, 1)

    def _compute_test_scores(self, X_train, X_test):
        """
        Calculates (or reutilizes from cache) the continuous discord scores of
        X_test with respect to X_train. Called internally by predict_blind and
        score_blind -- avoids running STUMPY twice in the same fold.
        """
        signature = (id(X_train), id(X_test), X_train.shape, X_test.shape)
        if self._last_test_signature == signature and self._last_scores_test is not None:
            return self._last_scores_test
 
        # Establish normal 'baseline' (recalculates threshold_ and concat_z_train)
        self.fit_predict(X_train)
 
        concat_z_test, trial_map_test = self._build_concat(X_test)
 
        # STUMPY A/B: search anomalies in the Test set (T_A) using the Training set (T_B) as a normal baseline
        mp = stumpy.stump(T_A=concat_z_test, m=self.m, T_B=self.concat_z_train, ignore_trivial=False, normalize=self.normalize)
        mp_distances = np.asarray(mp[:, 0], dtype=float)
        mp_distances[np.isinf(mp_distances)] = 0.0
 
        scores = self._get_scores(mp_distances, trial_map_test)
 
        self._last_scores_test = scores
        self._last_test_signature = signature
        return scores
    
    def predict_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """Test the new trajectories by comparing them ONLY with the training data."""
        scores = self._compute_test_scores(X_train, X_test)
        return np.where(scores > self.threshold_, -1, 1)

    def score_blind(self, X_train, X_test, full_data_matrix=None, test_idx=None):
        """
        Returns the continuous Matrix Profile distances (discord scores) as
        the anomaly score. Higher distance = more anomalous. Reuses the
        cache set by predict_blind if it was already called for the same
        (X_train, X_test) pair in this fold -- avoids a second, expensive
        STUMPY run.
        """
        return self._compute_test_scores(X_train, X_test)