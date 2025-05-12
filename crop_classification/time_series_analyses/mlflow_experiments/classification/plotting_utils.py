""" 
Plotting utilities for mlflow classification experiments
"""
from typing import Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.figure import Figure

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

def calculate_ROC(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ovr_clf: BaseEstimator
) -> Union[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    """ 
    This function calculates ROC curve for each class label in the dataset.

    Args: (i) X_train - training set data 
          (ii) y_train - training set labels
          (iii) X_val - validation set data
          (iv) y_val - validation set labels
          (v) ovr_clf - One-vs-Rest classifier

    Returns: (i) FPR_dict - a dictionary containing FPR values for each class
             (ii) TPR_dict - a dictionary containing TPR values for each class
             (iii) AUC_dict - a dictionary containing AUC for each class
    """
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    if isinstance(y_val, pd.Series):
        y_val = y_val.values

    # Binarize the training and validation set labels
    y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
    y_val_bin = label_binarize(y_val, classes=np.unique(y_train))
    n_classes = len(np.unique(y_train))

    ovr_clf.fit(X_train, y_train_bin)
    y_score = ovr_clf.predict_proba(X_val)

    # Store FPR, TPR and AUC for all classes in a dictionary
    FPR_dict = dict()
    TPR_dict = dict()
    AUC_dict = dict()
    FPR_GRID = np.linspace(0, 1, 100) # FPR grid for interpolation
    for i in range(n_classes):
        FPR, TPR, _ = roc_curve(y_val_bin[:, i], y_score[:, i])
        AUC_score = auc(FPR, TPR)

        """ 
        The lengths of FPR and TPR during different cv folds will not
        be the same (in general). Hence, the best way is to interpolate
        TPR along a uniform FPR grid.
        """
        TPR_interp = np.interp(FPR_GRID, FPR, TPR)
        TPR_interp[0] = 0.0

        FPR_dict[i] = FPR_GRID
        TPR_dict[i] = TPR_interp
        AUC_dict[i] = AUC_score

    # Returns the interpolated values
    return FPR_dict, TPR_dict, AUC_dict 

def calculate_PR(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        ovr_clf: BaseEstimator
) -> Union[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    if isinstance(y_val, pd.Series):
        y_val = y_val.values

    # Binarize the training and validation set labels
    y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
    y_val_bin = label_binarize(y_val, classes=np.unique(y_train))
    n_classes = len(np.unique(y_train))

    ovr_clf.fit(X_train, y_train_bin)
    y_score = ovr_clf.predict_proba(X_val)

    # Store precision, recall and thresholds in dictionaries
    precision_k = dict()
    recall_k = dict()
    threshold_k = dict()

    for i in range(n_classes):
        precision_k[i], recall_k[i], threshold_k[i] = precision_recall_curve(y_val_bin[:, i], y_score[:, i])
    
    return precision_k, recall_k, threshold_k

def plot_ROC(
        FPR: dict[int, list[np.ndarray]],
        TPR: dict[int, list[np.ndarray]],
        AUC: dict[int, list[float]],
        n_classes: int
) -> Figure:
    """ 
    This function plots the interpolated ROC curve.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        mean_FPR = np.mean(FPR[i], axis=0)
        mean_TPR = np.mean(TPR[i], axis=0)
        mean_AUC = np.mean(AUC[i])

        ax.plot(mean_FPR, mean_TPR, marker='.', linewidth=1.5, label=f"class {i}; AUC={mean_AUC:.2f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_title("ROC Curve", fontsize=15)
    ax.legend(loc="lower right", fontsize=12)

    return fig

def plot_PR(
        precision: dict[int, list[np.ndarray]],
        recall: dict[int, list[np.ndarray]],
        n_classes: int,
        n_points: int=100
) -> Figure:
    """
    This function plots the interpolated PR curve. 
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    recall_grid = np.linspace(0, 1, n_points)

    for i in range(n_classes):
        if len(precision[i]) == 0 or len(recall[i]) == 0:
            continue  # skip if no data

        interpolated_precisions = []

        for prec, rec in zip(precision[i], recall[i]):
            # Ensure recall is increasing for interpolation
            rec, prec = np.array(rec), np.array(prec)
            if rec[0] > rec[-1]:
                rec = rec[::-1]
                prec = prec[::-1]

            # Interpolate precision onto common recall grid
            interp_prec = np.interp(recall_grid, rec, prec)
            interpolated_precisions.append(interp_prec)

        interpolated_precisions = np.array(interpolated_precisions)
        mean_precision = np.mean(interpolated_precisions, axis=0)
        mean_auc = auc(recall_grid, mean_precision)

        ax.step(recall_grid, mean_precision, where="post", linewidth=2, label=f"Class {i}: AUC={mean_auc:.3f}")

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Mean Precision-Recall Curves (per class)", fontsize=15)
    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True)

    return fig

def plot_confusion_matrix(CM: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(CM, annot=True, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted", fontsize=15)
    ax.set_ylabel("True", fontsize=15)
    ax.set_title("Confusion Matrix", fontsize=15)

    return fig

def plot_learning_curve(
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        model_name: str,
        save: bool=True,
) -> Figure:
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(train_sizes, train_scores_mean, s=30, c="deepskyblue", edgecolor="k", label="Training")
    ax.plot(train_sizes, train_scores_mean, linewidth=2, color="deepskyblue")

    ax.scatter(train_sizes, val_scores_mean, s=30, c="indianred", edgecolor="k", label="Validation")
    ax.plot(train_sizes, val_scores_mean, linewidth=2, color="indianred")

    ax.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.5,
        color="deepskyblue"
    )

    ax.fill_between(
        train_sizes,
        val_scores_mean - val_scores_std,
        val_scores_mean + val_scores_std,
        alpha=0.5,
        color="indianred"
    )

    ax.set_xlabel("Training Set Size", fontsize=15)
    ax.set_ylabel("Accuracy", fontsize=15)
    ax.set_title(f"Learning Curve: {model_name}", fontsize=15)
    ax.legend(loc="lower right", fontsize=12)

    if save:
        plt.savefig(f"Learning_curve_{model_name}.png")

    return fig
    