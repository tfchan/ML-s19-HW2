"""Module of naive Bayes."""
import numpy as np


class NBClassifier:
    """Naive Bayes classifier."""

    def __init__(self, mode=0):
        """Initialize classifier with mode."""
        self._mode = mode
        self._prior = {}

    def _compute_prior(self, targets):
        """Compute prior probabilty for each class."""
        category, counts = np.unique(targets, return_counts=True)
        counts = zip(counts, counts / counts.sum())
        self._prior = dict(zip(category, counts))

    def fit(self, features, targets):
        """Fit the classifier with features and targets."""
        self._compute_prior(targets)
