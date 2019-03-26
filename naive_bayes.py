"""Module of naive Bayes."""
import numpy as np


class NaiveBayes:
    """Base class for naive Bayes classifier."""

    def __init__(self):
        """Initialize member variables."""
        self._classes = None
        self._class_count = None
        self._class_prior = None
        self._class_likelihood = None

    def _compute_prior(self, targets):
        """Compute prior probabilty for each class."""
        self._classes, self._class_count = np.unique(
            targets, return_counts=True)
        self._class_prior = self._class_count / self._class_count.sum()

    def fit(self, features, targets):
        """Fit the classifier with features and targets."""
        self._compute_prior(targets)
        self._compute_likelihood(features, targets)

    def predict_log_proba(self, features):
        """Give normalise log probability for each class of each sample."""
        jlp = self._joint_log_proba(features)
        result = []
        for i in range(jlp.shape[0]):
            jlp[i] = jlp[i] / jlp[i].sum()
            result += [dict(zip(self._classes, jlp[i]))]
        return result

    def get_imaginations(self):
        """Get imaginary features for each class."""
        ifs = self._imaginary_features()
        return dict(zip(self._classes, ifs))


class DiscreteNB(NaiveBayes):
    """Naive Bayes classifier for discrete features."""

    _bin_len = 8
    _n_bin = 256 // _bin_len

    def _compute_likelihood(self, features, targets):
        """Compute likelihood for discrete values."""
        features = features // self._bin_len
        n_class = self._classes.shape[0]
        n_feature = features.shape[1]
        self._class_likelihood = np.zeros((n_class, n_feature, self._n_bin))
        for c in range(n_class):
            class_features = features[targets == self._classes[c]]
            for i in range(n_feature):
                bin_freq_i = np.bincount(class_features[:, i],
                                         minlength=self._n_bin)
                bin_proba_i = bin_freq_i / bin_freq_i.sum()
                bin_proba_i[bin_proba_i == 0] = np.nan
                self._class_likelihood[c, i] = bin_proba_i

    def _joint_log_proba(self, features):
        """Compute log(P(c)P(x|c)) for each sample."""
        features = features // self._bin_len
        n_class = self._classes.shape[0]
        n_sample = features.shape[0]
        n_feature = features.shape[1]
        pc = np.log(self._class_prior)
        probas = np.zeros((n_sample, n_class))
        for i in range(n_sample):
            pxc = np.zeros(n_class)
            for c in range(n_class):
                pxc_c = np.array([self._class_likelihood[c, f, features[i, f]]
                                  for f in range(n_feature)])
                pxc_c[np.isnan(pxc_c)] = np.nanmin(pxc_c)
                pxc[c] = np.sum(np.log(pxc_c))
            probas[i] = pc + pxc
        return probas

    def _imaginary_features(self):
        """Compute imaginary features for each class."""
        split_point = self._n_bin // 2 * self._bin_len
        n_class = self._classes.shape[0]
        n_feature = self._class_likelihood.shape[1]
        class_imaginations = np.zeros((n_class, n_feature), dtype=int)
        for c in range(n_class):
            for f in range(n_feature):
                white_range = np.arange(split_point) // self._bin_len
                black_range = white_range + self._n_bin // 2
                white_proba = self._class_likelihood[c, f, white_range]
                black_proba = self._class_likelihood[c, f, black_range]
                is_black = np.nansum(black_proba) >= np.nansum(white_proba)
                class_imaginations[c, f] = int(is_black)
        return class_imaginations


class GussianNB(NaiveBayes):
    """Gussian naive Bayes classifier."""

    def _compute_likelihood(self, features, targets):
        """Compute likelihood using Gussian."""
        n_class = self._classes.shape[0]
        n_feature = features.shape[1]
        self._class_likelihood = np.zeros((n_class, n_feature, 2))
        for c in range(n_class):
            class_features = features[targets == self._classes[c]]
            self._class_likelihood[c, :, 0] = class_features.mean(axis=0)
            self._class_likelihood[c, :, 1] = class_features.var(axis=0)
