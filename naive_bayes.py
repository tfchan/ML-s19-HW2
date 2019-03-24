"""Module of naive Bayes."""
import numpy as np


class NBClassifier:
    """Naive Bayes classifier."""

    def __init__(self, mode=0):
        """Initialize classifier with mode."""
        self._mode = mode
        self._prior = {}
        self._likelihood = {}

    def _compute_prior(self, targets):
        """Compute prior probabilty for each class."""
        category, counts = np.unique(targets, return_counts=True)
        counts = zip(counts, counts / counts.sum())
        self._prior = dict(zip(category, counts))

    def _compute_discrete_likelihood(self, features, targets):
        """Compute likelihood for discrete values."""
        for category in self._prior.keys():
            category_features = features[targets == category]
            values_freq = []
            for col_num in range(features.shape[1]):
                value, counts = np.unique(category_features[:, col_num],
                                          return_counts=True)
                values_freq += [dict(zip(value, counts / counts.sum()))]
            self._likelihood[category] = values_freq

    def fit(self, features, targets):
        """Fit the classifier with features and targets."""
        self._compute_prior(targets)
        if self._mode == 0:
            self._compute_discrete_likelihood(features // 8, targets)
