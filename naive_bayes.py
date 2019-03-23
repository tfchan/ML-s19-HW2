"""Module of naive Bayes."""


class NBClassifier:
    """Naive Bayes classifier."""

    def __init__(self, mode=0):
        """Initialize classifier with mode."""
        self._mode = mode
