import csv
from datetime import datetime
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import requests
import openai
import clip
import seaborn as sns
import numpy as np
from typing import List, Optional, Any
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



def evaluate_zero_shot_predictions(
    labels: List[str],
    label_counts: List[int],
    test_images: List[torch.Tensor],
    test_image_filenames: List[str],
    model: torch.nn.Module,
    device: torch.device,
    save_confusion_matrix: bool = False,
    cm_title: str = "Confusion Matrix for Zero Shot Classification",
    short_labels: Optional[List[str]] = None,
    cm_file_path: Optional[str] = None,
    cm_file_name: str = "confusion_matrix.png"
) -> pd.DataFrame:
    """
    Evaluate zero-shot predictions using preprocessed test images and a CLIP model.

    Parameters:
    - labels (List[str]): Textual descriptions of the classes.
    - label_counts (List[int]): List of counts for each label in the same order as `labels`.
    - test_images (List[torch.Tensor]): List of preprocessed test image tensors.
    - test_image_filenames (List[str]): List of filenames corresponding to test images.
    - model (torch.nn.Module): CLIP model for evaluation.
    - device (torch.device): Device for model and tensor computations.
    - save_confusion_matrix (bool): Whether to save the confusion matrix plot.
    - cm_title (str): Title for the confusion matrix plot.
    - short_labels (Optional[List[str]]): Short labels for the confusion matrix plot.
    - cm_file_path (Optional[str]): Path to save the confusion matrix image.
    - cm_file_name (str): Filename for the confusion matrix image.

    Returns:
    - pd.DataFrame: DataFrame with classification metrics and probabilities for each label.
    """
    # Tokenize and encode labels using the CLIP model
    text_inputs = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)  # Normalize embeddings

    predicted_labels = []
    results_data = []

    # Generate true labels
    true_labels = [label for label, count in enumerate(label_counts) for _ in range(count)]

    # Process preloaded and preprocessed test image tensors
    for idx, image_tensor in enumerate(test_images):
        with torch.no_grad():
            # Compute image embedding
            image_features = model.encode_image(image_tensor.to(device))
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity scores and probabilities
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            predicted_label = similarity.argmax().item()
            probabilities = similarity.squeeze().tolist()

            predicted_labels.append(predicted_label)

            # Store result data for each image
            results_data.append({
                "image_filename": test_image_filenames[idx],
                "true_label": true_labels[idx],
                "predicted_label": predicted_label,
                **{f"prob_{label}": prob for label, prob in zip(labels, probabilities)}
            })

    # Create a DataFrame for results
    results_df = pd.DataFrame(results_data)

    # Calculate confusion matrix
    cm = confusion_matrix(
        [result["true_label"] for result in results_data],
        [result["predicted_label"] for result in results_data],
        labels=list(range(len(labels)))
    )

    # Extract TN, FP, FN, TP for binary classification
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=1, average="binary")
    recall = recall_score(true_labels, predicted_labels, pos_label=1, average="binary")
    f1 = f1_score(true_labels, predicted_labels, pos_label=1, average="binary")

    # Sensitivity and Specificity
    sensitivity = recall  # Sensitivity is also known as recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Matthews Correlation Coefficient (MCC)
    mcc = ((tp * tn) - (fp * fn)) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5) if (tp + fp) > 0 and (tp + fn) > 0 and (tn + fp) > 0 and (tn + fn) > 0 else 0

    # False Positive Rate (FPR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Area Under Curve (AUC)
    if f"prob_{labels[1]}" in results_df.columns:
        probabilities = results_df[f"prob_{labels[1]}"].tolist()
        auc = roc_auc_score([1 if label == 1 else 0 for label in true_labels], probabilities)
    else:
        auc = float('nan')

    # Compile metrics into a DataFrame
    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Sensitivity (Recall)",
            "Specificity",
            "Precision",
            "F1 Score",
            "AUC",
            "Matthews Correlation Coefficient (MCC)",
            "False Positive Rate (FPR)"
        ],
        "Value": [
            accuracy,
            sensitivity,
            specificity,
            precision,
            f1,
            auc,
            mcc,
            fpr
        ]
    })

    # Plot confusion matrix
    if short_labels is None:
        short_labels = labels

    if save_confusion_matrix:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short_labels, yticklabels=short_labels)
        plt.title(cm_title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        if cm_file_path:
            os.makedirs(cm_file_path, exist_ok=True)
            cm_save_path = os.path.join(cm_file_path, cm_file_name)
            plt.savefig(cm_save_path)
            plt.close()
        else:
            plt.show()

    return metrics, results_df



def few_shot_fault_classification(
    test_images: List[Image.Image],
    test_image_filenames: List[str],
    nominal_images: List[Image.Image],
    nominal_descriptions: List[str],
    defective_images: List[Image.Image],
    defective_descriptions: List[str],
    num_few_shot_nominal_imgs: int,
    model: torch.nn.Module,
    file_path: str = '/content',
    file_name: str = 'image_classification_results.csv',
    print_one_liner: bool = False
):
    """
    Classify test images as nominal or defective based on similarity to nominal and defective images.

    Parameters:
    - test_images (List[Image.Image]): List of test images to classify.
    - test_image_filenames (List[str]): Corresponding filenames for test images.
    - nominal_images (List[Image.Image]): List of nominal reference images.
    - nominal_descriptions (List[str]): Descriptions for nominal images.
    - defective_images (List[Image.Image]): List of defective reference images.
    - defective_descriptions (List[str]): Descriptions for defective images.
    - num_few_shot_nominal_imgs (int): Number of nominal images used in few shot learning.
    - file_path (str): Directory path where the results CSV will be saved.
    - file_name (str): Name of the CSV file.
    - print_one_liner (bool): Whether to print classification summaries.

    Returns:
    - List[Dict[str, Any]]: List of classification results for each test image.
    """

    # Ensure inputs are lists
    if not isinstance(test_images, list):
        test_images = [test_images]
    if not isinstance(test_image_filenames, list):
        test_image_filenames = [test_image_filenames]
    if not isinstance(nominal_images, list):
        nominal_images = [nominal_images]
    if not isinstance(nominal_descriptions, list):
        nominal_descriptions = [nominal_descriptions]
    if not isinstance(defective_images, list):
        defective_images = [defective_images]
    if not isinstance(defective_descriptions, list):
        defective_descriptions = [defective_descriptions]

    # Prepare full path for the CSV file
    csv_file = os.path.join(file_path, file_name)
    results = []

    with torch.no_grad():
        # Encode nominal images
        nominal_features = torch.stack([model.encode_image(nominal_img) for nominal_img in nominal_images])
        nominal_features /= nominal_features.norm(dim=-1, keepdim=True)

        # Encode defective images
        defective_features = torch.stack([model.encode_image(defective_img) for defective_img in defective_images])
        defective_features /= defective_features.norm(dim=-1, keepdim=True)

        # Prepare list to save data for CSV
        csv_data = []

        # Process each test image
        for idx, test_img in enumerate(test_images):
            test_features = model.encode_image(test_img)
            test_features /= test_features.norm(dim=-1, keepdim=True)

            # Initialize variables to store max similarities and indices
            max_nominal_similarity = -float('inf')
            max_defective_similarity = -float('inf')
            max_nominal_idx = -1
            max_defective_idx = -1

            # Loop through each nominal image to find max similarity
            for i in range(nominal_features.shape[0]):
                similarity = (test_features @ nominal_features[i].T).item()
                if similarity > max_nominal_similarity:
                    max_nominal_similarity = similarity
                    max_nominal_idx = i

            # Loop through each defective image to find max similarity
            for j in range(defective_features.shape[0]):
                similarity = (test_features @ defective_features[j].T).item()
                if similarity > max_defective_similarity:
                    max_defective_similarity = similarity
                    max_defective_idx = j

            # Convert similarities to probabilities
            similarities = torch.tensor([max_nominal_similarity, max_defective_similarity])
            probabilities = F.softmax(similarities, dim=0).tolist()
            prob_not_defective = probabilities[0]
            prob_defective = probabilities[1]

            # Determine classification result
            classification = "Defective" if prob_defective > prob_not_defective else "Nominal"

            # Append result for CSV, including matched nominal and defective descriptions
            csv_data.append({
                "datetime_of_operation": datetime.now().isoformat(),
                "num_few_shot_nominal_imgs": num_few_shot_nominal_imgs,
                "image_path": test_image_filenames[idx],
                "image_name": test_image_filenames[idx].split('/')[-1],
                "classification_result": classification,
                "non_defect_prob": round(prob_not_defective, 3),
                "defect_prob": round(prob_defective, 3),
                "nominal_description": nominal_descriptions[max_nominal_idx],
                "defective_description": defective_descriptions[max_defective_idx] if defective_images else "N/A"
            })

            # Optionally print one-liner summary for each test image
            if print_one_liner:
                print(f"{test_image_filenames[idx]} classified as {classification} "
                      f"(Nominal Prob: {prob_not_defective:.3f}, Defective Prob: {prob_defective:.3f})")

    # Write to CSV (append mode if file exists, write mode if not)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a' if file_exists else 'w', newline='') as file:
        fieldnames = [
            "datetime_of_operation", "num_few_shot_nominal_imgs", "image_path", "image_name",
            "classification_result", "non_defect_prob", "defect_prob", "nominal_description", "defective_description"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header if file doesn't exist
        if not file_exists:
            writer.writeheader()

        # Write each row of data
        for row in csv_data:
            writer.writerow(row)

    return ""
