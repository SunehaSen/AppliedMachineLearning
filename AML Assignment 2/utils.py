from typing import Tuple

import re
import string
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt_tab')

import warnings
warnings.filterwarnings('ignore')

def plot_distribution(
        series: pd.Series,
        axs: plt.Axes,
    ) -> None:
    """
    Plot the distribution of a categorical variable.

    Parameters
    ----------
    series : pd.Series
        The input Series.
    axs : plt.Axes
        The matplotlib axes object.
    """

    sns.countplot(x=series,
                  order=series.value_counts(ascending=False).index,
                  ax=axs)
    abs_values = series.value_counts(ascending=False).values
    axs.bar_label(container=axs.containers[0], labels=abs_values)

def process_text(
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'label',
    ) -> pd.DataFrame:
    """
    Preprocess the text data by converting to lowercase, tokenizing, removing special characters, stop words, punctuation,
    stemming, and encoding the labels.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    text_col : str
        The name of the text column.
    label_col : str
        The name of the label column.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame.
    """

    # Convert the text to lowercase
    df[f'processed_{text_col}'] = df[text_col].str.lower()
    # Tokenization
    df[f'processed_{text_col}'] = df[f'processed_{text_col}'].apply(word_tokenize)

    # Removing special characters
    df[f'processed_{text_col}'] = df[f'processed_{text_col}'].apply(lambda x: [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in x])

    # Removing stop words and punctuation
    stop_words = set(stopwords.words('english'))
    df[f'processed_{text_col}'] = df[f'processed_{text_col}'].apply(lambda x: [word for word in x if word not in stop_words and word not in string.punctuation])

    # Stemming
    ps = PorterStemmer()
    df[f'processed_{text_col}'] = df[f'processed_{text_col}'].apply(lambda x: [ps.stem(word) for word in x])

    # Convert the preprocessed text back to string
    df[f'processed_{text_col}'] = df[f'processed_{text_col}'].apply(lambda x: ' '.join(x))

    # Encode the labels
    df[label_col] = df[label_col].map({'ham': 0, 'spam': 1})

    # Drop the original text column
    df = df.drop(columns=[text_col])

    return df

def split_train_test_valid(
        df: pd.DataFrame,
        text_col: str = 'processed_text',
        label_col: str = 'label',
        test_size: float = 0.2,
        random_state: int = 1,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """ 
    Split the dataset into training, testing, and validation sets.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    text_col : str
        The name of the text column.
    label_col : str
        The name of the label column.
    test_size : float
        The proportion of the dataset to include in the test split.
    random_state : int
        The seed used by the random number generator.

    Returns
    -------
    Tuple(pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series)
        The training, testing, and validation sets.
    """

    X, y = df[text_col], df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

def print_report(
        pred: pd.Series,
        gt: pd.Series,
    ) -> None:
    """
    Print the classification report.

    Parameters
    ----------
    pred : pd.Series
        The predicted values.
    gt : pd.Series
        The ground truth values.
    """ 

    print('Accuracy:', accuracy_score(gt, pred))
    print('Precision:', precision_score(gt, pred))
    print('Recall:', recall_score(gt, pred))
    print('F1:', f1_score(gt, pred))
    print('AUCPR:', auc(*precision_recall_curve(gt, pred)[:2]))

def track_using_mlflow(
        model,
        name,
        test_X,
        test_y,
):
    with mlflow.start_run(run_name=name):
        y_pred = model.predict(test_X)
        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", accuracy_score(test_y, y_pred))
        mlflow.log_metric("precision", precision_score(test_y, y_pred))
        mlflow.log_metric("recall", recall_score(test_y, y_pred))
        mlflow.log_metric("f1 score", f1_score(test_y, y_pred))
        mlflow.log_metric("AUCPR", auc(*precision_recall_curve(test_y, y_pred)[:2]))
        mlflow.sklearn.log_model(model, "model")
        
        tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn-model",
                registered_model_name=f"{name} model"
        )
        if tracking_url_type != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name=name)
        else:
                mlflow.sklearn.log_model(model, "model")