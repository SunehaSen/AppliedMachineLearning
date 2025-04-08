import numpy as np
from sklearn.base import BaseEstimator

def score(text: str, model: BaseEstimator, threshold: float) -> tuple[bool, float]:

    # Ensure text is a list, as scikit-learn models expect 2D input
    text_vectorized = np.array([text])

    # Predict probability (assumes binary classification with predict_proba)
    propensity = model.predict_proba(text_vectorized)[0, 1]

    # Apply threshold to get prediction
    prediction = propensity >= threshold

    return prediction, propensity