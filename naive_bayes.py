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
                self._class_likelihood[c][i] = bin_proba_i

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

    def get_imagination(self):
        """Return imaginary features of each class."""
        split_point = 256 // 2
        class_imagination = {}
        for class_ in self._class_prior.keys():
            img_features = []
            for feature in range(len(self._class_likelihood[class_])):
                class_feature_likeli = self._class_likelihood[class_][feature]
                white_proba = [class_feature_likeli.get(
                    i // _bin_len, 0) for i in range(split_point)]
                black_proba = [class_feature_likeli.get(
                    (i + split_point) // _bin_len, 0) for i in range(split_point)]
                is_black = 1 if sum(black_proba) >= sum(white_proba) else 0
                img_features += [is_black]
            class_imagination[class_] = img_features
        return class_imagination


class GussianNB(NaiveBayes):
    """Gussian naive Bayes classifier."""

    def _compute_likelihood(self, features, targets):
        """Compute likelihood using Gussian."""

    def fit(self, features, targets):
        """Fit the classifier with features and targets."""
        self._compute_prior(targets)
        self._compute_likelihood(features, targets)
