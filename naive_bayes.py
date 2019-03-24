"""Module of naive Bayes."""
import numpy as np


class NaiveBayes:
    """Base class for naive Bayes classifier."""

    def __init__(self):
        """Initialize member variables."""
        self._prior = {}
        self._likelihood = {}

    def _compute_prior(self, targets):
        """Compute prior probabilty for each class."""
        category, counts = np.unique(targets, return_counts=True)
        counts = zip(counts, counts / counts.sum())
        self._prior = dict(zip(category, counts))


class DiscreteNB(NaiveBayes):
    """Naive Bayes classifier for discrete features."""

    def _compute_likelihood(self, features, targets):
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
        self._compute_likelihood(features // 8, targets)
