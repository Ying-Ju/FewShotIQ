import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from typing import List, Optional, Dict


def compute_classification_metrics(
    csv_file: str,
    labels: List[str],
    label_counts: List[int],
    num_few_shot_nominal_imgs: int,
    save_confusion_matrix: bool = False,
    cm_file_path: Optional[str] = None,
    cm_file_name: str = "confusion_matrix.png"
) -> pd.DataFrame:
    """
    Compute classification metrics from a CSV file containing classification results.

    Parameters:
    - csv_file (str): Path to the CSV file with classification results.
    - labels (List[str]): List of label names.
    - label_counts (List[int]): List of counts for each label in `labels`, in the same order.
    - num_few_shot_nominal_imgs (int): Number of nominal images used for learning.
    - save_confusion_matrix (bool, optional): If True, saves the confusion matrix plot as an image. Default is False.
    - cm_file_path (Optional[str], optional): Directory to save the confusion matrix image if `save_confusion_matrix` is True.
    - cm_file_name (str, optional): Filename for the confusion matrix image. Default is "confusion_matrix.png".

    Returns:
    - pd.DataFrame: DataFrame with classification metrics.

    Raises:
    - ValueError: If label counts do not match the number of labels or if they don't sum to the total number of records in the CSV.
    """
    # Load the classification results from the CSV file
    results_df = pd.read_csv(csv_file)

    # Verify label counts
    total_records = results_df.shape[0]
    if len(labels) != len(label_counts) or sum(label_counts) != total_records:
        raise ValueError("Label counts must match the number of labels and sum to the total records in the CSV file.")

    # Generate true labels based on counts
    true_labels = []
    for label, count in zip(labels, label_counts):
        true_labels.extend([label] * count)

    # Extract predicted labels from the CSV
    predicted_labels = results_df["classification_result"].tolist()

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)  # Supports binary classification

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=labels[1], average="binary")
    recall = recall_score(true_labels, predicted_labels, pos_label=labels[1], average="binary")
    f1 = f1_score(true_labels, predicted_labels, pos_label=labels[1], average="binary")

    # Sensitivity and Specificity
    sensitivity = recall  # Sensitivity is also known as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # AUC (Area Under Curve)
    # Using defect probabilities for AUC if available
    if "defect_prob" in results_df.columns:
        probabilities = results_df["defect_prob"].tolist()
        auc = roc_auc_score([1 if label == labels[1] else 0 for label in true_labels], probabilities)
    else:
        auc = float('nan')

    # Compile metrics into a DataFrame
    metrics = pd.DataFrame({
        "Metric": ["Nominal Learning Size", "Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score", "AUC"],
        "Value": [num_few_shot_nominal_imgs, accuracy, sensitivity, specificity, precision, f1, auc]
    })

    # Plot confusion matrix
    total_cases = sum(label_counts)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=max(label_counts))
    plt.title(f"Confusion Matrix Based on a Nominal Learning Set Size of {num_few_shot_nominal_imgs}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Save the confusion matrix plot if specified
    if save_confusion_matrix:
        if cm_file_path:
            os.makedirs(cm_file_path, exist_ok=True)
            cm_save_path = os.path.join(cm_file_path, cm_file_name)
        else:
            cm_save_path = cm_file_name
        plt.savefig(cm_save_path)
        plt.close()
    else:
        plt.show()

    return metrics
