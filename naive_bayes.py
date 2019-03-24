"""Module of naive Bayes."""
import numpy as np


class NaiveBayes:
    """Base class for naive Bayes classifier."""

    def __init__(self):
        """Initialize member variables."""
        self._class_count = {}
        self._class_prior = {}
        self._class_likelihood = {}

    def _compute_prior(self, targets):
        """Compute prior probabilty for each class."""
        category, counts = np.unique(targets, return_counts=True)
        self._class_count = dict(zip(category, counts))
        self._class_prior = dict(zip(category, counts / counts.sum()))


class DiscreteNB(NaiveBayes):
    """Naive Bayes classifier for discrete features."""

    def _compute_likelihood(self, features, targets):
        """Compute likelihood for discrete values."""
        for category in self._class_prior.keys():
            category_features = features[targets == category]
            values_freq = []
            for col_num in range(features.shape[1]):
                value, counts = np.unique(category_features[:, col_num],
                                          return_counts=True)
                values_freq += [dict(zip(value, counts / counts.sum()))]
            self._class_likelihood[category] = values_freq

    def _preprocess_features(self, features):
        return features // 8

    def fit(self, features, targets):
        """Fit the classifier with features and targets."""
        self._compute_prior(targets)
        self._compute_likelihood(self._preprocess_features(features), targets)

    def predict_log_proba(self, features):
        """Give log probability for each class of each sample."""
        new_features = self._preprocess_features(features)
        sample_posteriors = []
        for sample in new_features:
            class_posterior = {}
            # Calculate P(x|c) * P(c) in log scale
            for class_ in self._class_prior.keys():
                likelihoods = np.array([self._class_likelihood[class_][i].get(
                    sample[i], np.nan) for i in range(sample.shape[0])])
                likelihoods[np.isnan(likelihoods)] = np.nanmin(likelihoods)
                class_posterior[class_] = -np.sum(np.log(likelihoods))
                class_posterior[class_] -= np.log(self._class_prior[class_])
            # Normalise to sum up to 1
            total = sum(class_posterior.values())
            for class_ in self._class_prior.keys():
                class_posterior[class_] /= total
            sample_posteriors += [class_posterior]
        return sample_posteriors
