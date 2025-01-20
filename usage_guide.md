# FewShotIQ: Usage Guide

FewShotIQ is a Python package for efficient few-shot image classification and visualization. This guide provides detailed instructions and examples for using the package's functions.

---

## Installation

Install the package via pip:

```bash
pip install FewShotIQ
```

## Overview of Modules and Functions

`image_utils` Module

This module provides functions to handle images, such as loading, animating, and working with datasets.

- `load_image(image_path_or_url: str)`
    Loads an image from a local path or URL. Returns a PIL.Image.Image object.

- `animate_images(...)`
    Creates side-by-side animations of nominal and defective images. Optionally saves the animation as a GIF.

- `load_package_image(image_path: str)`
    Loads an image from the package's data directory.


`classification` Module
Perform zero-shot and few-shot classification with pre-trained models.

- `evaluate_zero_shot_predictions(...)`
    Classifies an image using text prompts.

- `few_shot_fault_classification(...)`
    Fine-tunes a model for few-shot classification tasks.


`evaluation` Module
Evaluate classification performance.

- `compute_classification_metrics(...)`
    Compute standard classification metrics such as accuracy, precision, recall, and F1-score from a CSV file containing classification results.

- `map_multiclass_labels(row, major_labels, sub_labels)`
    Map classification_result and defective_description to multiclass labels

- `calculate_specificity(conf_matrix, labels)`
   Calculate specificity for each class


`visualization` Module
Visualize the effects of varying few-shot learning size or model parameters.

- `vary_number_fewshot_example(...)`
Vary the size of the learning set and evaluate classification performance.

- `create_confusion_matrix_gif(...)`
Creates a GIF of confusion matrix images and displays it.

- `preprocess_and_plot_learning_size_metrics(...)`
Preprocesses raw repeated-block data and plots learning size metrics.

- `preprocess_and_plot_two_models(...)`
Preprocesses two raw datasets and plots learning size metrics for both.

## Detailed Usage Examples

### `image_utils` Module
#### Load an image from the package's data folder

```python
from FewShotIQ.image_utils import load_package_image

image = load_package_image("pan_images/test/defective/IMG_1514.JPG")
image.show()
```

#### Loading Images 

```python
from FewShotIQ.image_utils import load_image

# Load an image from a local path
img = load_image("path/to/image.jpg")
img.show()

# Load an image from a URL
img = load_image("https://github.com/Ying-Ju/FewShotIQ/blob/main/FewShotIQ/data/pan_images/test/defective/IMG_1514.JPG?raw=true")
img.show()
```

#### Create a GIF of animated images

Animate nominal and defective images side-by-side with resizing only for images larger than 800x800 pixels.

```python
from FewShotIQ.image_utils import get_image_urls, animate_images

# Set the GitHub repository details
repo_owner = "fmegahed"
repo_name = "qe_genai"
base_path = "data/pan_images/train"

# Retrieve URLs for each subfolder
nominal_image_urls = get_image_urls(repo_owner, repo_name, base_path, "nominal")
defective_image_urls = get_image_urls(repo_owner, repo_name, base_path, "defective")

# Extract filenames for descriptions
nominal_image_filenames = [url.split('/')[-1] for url in nominal_image_urls]
defective_image_filenames = [url.split('/')[-1] for url in defective_image_urls]

# Descriptions for each nominal image
nominal_descriptions = [
    f"Image {filename}: A pan surface with no simulated cracks"
    for filename in nominal_image_filenames
]

# Fault descriptions for each defective image
defective_descriptions = [
    f"Image {filename}: A pan surface with a simulated crack"
    for filename in defective_image_filenames
]


animate_images(
    nominal_image_urls=nominal_image_urls,
    nominal_labels=nominal_descriptions,
    defective_image_urls=defective_image_urls,
    defective_labels=defective_descriptions,
    resize_factor=0.15,
    pause_time=2.0,
    save_fig=True,
    file_path=save_folder,
    file_name="exp01_learning_images.gif"
)
```

### `classification` Module

In order to use the functions `evaluate_zero_shot_predictions()` and `few_shot_fault_classification` in this module, we need to import necessary libraries, set the OpenAI API key, saved folder location, and load images first. The following code gives an example for this purpose. In the example below, users should replace `OPENAI_API_KEY` by their own key and replace `/content` by their save folder.


```python
import os
from io import BytesIO
import importlib.util
import subprocess
import csv
from urllib.parse import urlparse
from IPython.display import Image as IPyImage, display, clear_output
import time
import json
import random

# packages that need to be installed in the colab
packages = ['openai', 'torch', 'pillow', 'requests', 'clip', 'datetime', 'matplotlib', 'pandas', 'sklearn', 'seaborn', 'numpy', 'typing']

# installing the packages if needed
for package in packages:
  spec = importlib.util.find_spec(package)
  if spec is None:
    print(f"Installing {package}...")
    subprocess.check_call(['pip', 'install', package])
    if package == 'clip':  # Special case for CLIP from GitHub
      subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/CLIP.git'])
  else:
    print(f"{package} already installed.")

# importing those packages
import requests
from PIL import Image
import openai
import clip
import torch
import pandas as pd
import seaborn as sns
import numpy as np

# importing specific functions/modules from those libraries
from typing import List, Optional, Union, Any
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# package for importing our hidden key
openai.api_key = userdata.get('OPENAI_API_KEY') # insert your key here

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-L/14", device=device)

save_folder = '/content'

# Check if the default save folder exists
if not os.path.exists(save_folder):
    print(f"Folder '{save_folder}' does not exist.")
    # Prompt for a new folder path
    save_folder = input("Enter a new folder path: ").strip()
    os.makedirs(save_folder, exist_ok=True)

print(f"Final save folder: {save_folder}")

import random
from FewShotIQ.image_utils import get_image_urls, load_image


# Set the GitHub repository details
repo_owner = "fmegahed"
repo_name = "qe_genai"
base_path = "data/pan_images/train"

# Retrieve URLs for each subfolder
nominal_image_urls = get_image_urls(repo_owner, repo_name, base_path, "nominal")
defective_image_urls = get_image_urls(repo_owner, repo_name, base_path, "defective")

# Extract filenames for descriptions
nominal_image_filenames = [url.split('/')[-1] for url in nominal_image_urls]
defective_image_filenames = [url.split('/')[-1] for url in defective_image_urls]

# Descriptions for each nominal image
nominal_descriptions = [
    f"Image {filename}: A pan surface with no simulated cracks"
    for filename in nominal_image_filenames
]

# Fault descriptions for each defective image
defective_descriptions = [
    f"Image {filename}: A pan surface with a simulated crack"
    for filename in defective_image_filenames
]

# Load nominal (good) and faulty images from GitHub
nominal_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in nominal_image_urls]
defective_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in defective_image_urls]

# Load the test images
base_path = "data/pan_images/test"

test_nominal_image_urls = get_image_urls(repo_owner, repo_name, base_path, "nominal")
test_defective_image_urls = get_image_urls(repo_owner, repo_name, base_path, "defective")
test_image_urls = test_nominal_image_urls + test_defective_image_urls

test_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in test_image_urls]
```

#### Zero-Shot Classification with CLIP

Suppose users have imported libraries, set the OpenAI API Key, saved folder location, and loaded images. In that case, they can directly execute the `evaluate_zero_shot_predictions()` function for zero-shot classification with CLIP. Otherwise, please review the previous sections for the relevant topics. 

```python
from FewShotIQ.classification import evaluate_zero_shot_predictions

zero_shot_metrics, zero_shot_df = evaluate_zero_shot_predictions(
    labels=['A metallic pan free of black scuff marks', 'A metallic pan with a simulated scuff mark drawn by a black marker'],
    label_counts= [50,50],
    test_images=test_images,
    test_image_filenames = [url.split('/')[-1] for url in test_image_urls],
    model=model,
    device=device,
    save_confusion_matrix = True,
    cm_title = "Confusion Matrix for Zero Shot Classification for Experiment 01",
    short_labels=['Nominal', 'Defective'],
    cm_file_path = save_folder,  #save_folder should be replaced by the name of user's folder
    cm_file_name = "zero_shot_confusion_matrix.png"
)

print("\n\033[1mConfusion Matrix for Experiment 01\033[1m\n")
display(zero_shot_metrics.round(3).set_index("Metric").T)

```

#### Few-Shot Classification with CLIP

Suppose users have imported libraries, set the OpenAI API Key, saved folder location, and loaded images. In that case, they can directly execute the `few_shot_fault_classification()` function for few-shot classification with CLIP. Otherwise, please review the previous sections for the relevant topics. 

```python
from FewShotIQ.classification import few_shot_fault_classification

classification_results = few_shot_fault_classification(
    test_images = test_images,
    test_image_filenames = [url.split('/')[-1] for url in test_image_urls],
    nominal_images = nominal_images,
    nominal_descriptions = nominal_descriptions,
    defective_images = defective_images,
    defective_descriptions = defective_descriptions,
    num_few_shot_nominal_imgs = len(nominal_images),
    model = model,
    file_path = save_folder,      #save_folder should be replaced by the name of user's folder
    file_name = 'results.csv',   
    print_one_liner = False
)
```

### `evaluation` Module

### Computing Classification Metrics 
The following code chunk gives an example to compute standard classification metrics such as accuracy, precision, recall, and F1-score from a CSV file containing classification results. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 75 images for each defective class (Band, Bimodal, Single Crystal). Our example file `results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/results.csv?token=GHSAT0AAAAAAC5O5XFHU7F6M2YPM5RHTW3SZ4L6IYQ).


```python
import os
import pandas as pd                  # For working with dataframes
import matplotlib.pyplot as plt      # For plotting
import seaborn as sns                # For creating heatmaps
from sklearn.metrics import (        # For evaluating model performance
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from typing import List, Optional, Dict

from FewShotIQ.evaluation import compute_classification_matrics

classification_metrics = compute_classification_metrics(
    csv_file = f'{save_folder}/results.csv',
    labels = ["Nominal", "Defective"],
    label_counts = [len(test_nominal_images), len(test_defective_images)],
    num_few_shot_nominal_imgs = len(nominal_images),
    save_confusion_matrix = True,
    cm_file_path = save_folder,
    cm_file_name = "confusion_matrix.png"
    )

# Rounding for display
classification_metrics.Value = classification_metrics.Value.round(3)

# Drop the row corresponding to num_few_shot_nominal_imgs since it is not a classification metric
classification_metrics = classification_metrics.drop(index=0).reset_index(drop=True)

# Display the created confusion metrics figure
print("\n\033[1mConfusion Matrix\033[1m")
display(IPyImage(filename=f"{save_folder}/confusion_matrix.png"))

# Display the metrics as a table with three digit precision
print("\n\033[1mClassification Metrics:\033[1m")
display(classification_metrics)
```


#### Computing Major Label Accuracy for Multiclass

Here, we use an example to show the use of these two functions: `map_multiclass_labels` and `calculate_specificity()` to compute major label accuracy for multiclass. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 75 images for each defective class (Band, Bimodal, Single Crystal). Our example file `results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/results.csv?token=GHSAT0AAAAAAC5O5XFHU7F6M2YPM5RHTW3SZ4L6IYQ).

```python
import pandas as pd                  # For working with dataframes
import numpy as np                   # For numerical computations
import matplotlib.pyplot as plt      # For plotting
import seaborn as sns                # For creating heatmaps
from sklearn.metrics import (        # For evaluating model performance
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from FewShotIQ.evaluation import map_multiclass_labels, calculate_specificity

classification_results = pd.read_csv(f'{save_folder}/results.csv')

# Define the major and sub-labels
labels = ['Nominal', 'Band', 'Bimodal', 'Single Crystal']

# Apply the map_multiclass_labels function with additional arguments
classification_results['predicted_label'] = classification_results.apply(
    map_multiclass_labels, axis=1, major_labels=labels
)

# Create true labels based on the given sequence
true_labels = (
    ['Nominal'] * 225 +
    ['Band'] * 75 +
    ['Bimodal'] * 75 +
    ['Single Crystal'] * 75
)

classification_results['true_label'] = true_labels

# Compute the confusion matrix
cm = confusion_matrix(classification_results['true_label'], classification_results['predicted_label'], labels=labels)

# Compute metrics
accuracy = accuracy_score(classification_results['true_label'], classification_results['predicted_label'])
recall = recall_score(classification_results['true_label'], classification_results['predicted_label'], average='macro', zero_division=0)
precision = precision_score(classification_results['true_label'], classification_results['predicted_label'], average='macro', zero_division=0)
f1 = f1_score(classification_results['true_label'], classification_results['predicted_label'], average='macro', zero_division=0)
specificity = calculate_specificity(cm, labels)
average_specificity = np.mean(specificity)


# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=225)
plt.title("Confusion Matrix for Primary Multi-Class Labels for Experiment 05")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(f"{save_folder}/exp05_confusion_matrix_major_labels.png")
plt.show()

# Display metrics
classification_metrics = {
    'Accuracy': accuracy,
    'Recall (Macro Avg)': recall,
    'Specificity (Macro Avg)': average_specificity,
    'Precision (Macro Avg)': precision,
    'F1-Score (Macro Avg)': f1
}
print("\n\033[1mClassification Metrics:\033[1m")
classification_metrics = pd.DataFrame(classification_metrics.items(), columns=['Metric', 'Value'])
classification_metrics['Value'] = classification_metrics['Value'].round(3)
display(classification_metrics)
```   

#### Computing the Per Sub Label Accuracy

Similar to the previous example, we show the use of these two functions: `map_multiclass_labels` and `calculate_specificity()` to compute per sub label accuracy for multiclass. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 25 images for each defective class (Low Band, Medium Band, High Band, Low Bimodal, Medium Bimodal, High Bimodal, Low Single, Medium Single, High Single). Our example file `results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/results.csv?token=GHSAT0AAAAAAC5O5XFHU7F6M2YPM5RHTW3SZ4L6IYQ).


 ```python
import pandas as pd                  # For working with dataframes
import numpy as np                   # For numerical computations
import matplotlib.pyplot as plt      # For plotting
import seaborn as sns                # For creating heatmaps
from sklearn.metrics import (        # For evaluating model performance
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)
from FewShotIQ.evaluation import map_multiclass_labels, calculate_specificity

classification_results = pd.read_csv(f'{save_folder}/results.csv')

# Define the major and sub-labels
major_labels = ['Nominal', 'Band', 'Bimodal', 'Single']
sub_labels = ['Low', 'Medium', 'High']

# Apply the map_multiclass_labels function with additional arguments
classification_results['predicted_label'] = classification_results.apply(
    map_multiclass_labels, axis=1, major_labels=major_labels, sub_labels=sub_labels
)

# Create true labels based on the given sequence
true_labels = (
    ['Nominal'] * 225 +
    ['High Band'] * 25 +
    ['Low Band'] * 25 +
    ['Medium Band'] * 25 +
    ['High Bimodal'] * 25 +
    ['Low Bimodal'] * 25 +
    ['Medium Bimodal'] * 25 +
    ['High Single'] * 25 +
    ['Low Single'] * 25 +
    ['Medium Single'] * 25
)

classification_results['true_label'] = true_labels

# Compute the confusion matrix
labels = [
    'Nominal', 'High Band', 'Low Band', 'Medium Band',
    'High Bimodal', 'Low Bimodal', 'Medium Bimodal',
    'High Single', 'Low Single', 'Medium Single'
]

cm = confusion_matrix(classification_results['true_label'], classification_results['predicted_label'], labels=labels)

# Compute metrics
accuracy = accuracy_score(classification_results['true_label'], classification_results['predicted_label'])
recall = recall_score(classification_results['true_label'], classification_results['predicted_label'], average='macro', zero_division=0)
precision = precision_score(classification_results['true_label'], classification_results['predicted_label'], average='macro', zero_division=0)
f1 = f1_score(classification_results['true_label'], classification_results['predicted_label'], average='macro', zero_division=0)
specificity = calculate_specificity(cm, labels)
average_specificity = np.mean(specificity)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, vmin=0, vmax=225)
plt.title("Confusion Matrix for Secondary Multi-Class Labels for Experiment 05")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(f"{save_folder}/exp05_confusion_matrix_minor_labels.png")
plt.show()


# Display metrics
classification_metrics = {
    'Accuracy': accuracy,
    'Recall (Macro Avg)': recall,
    'Specificity (Macro Avg)': average_specificity,
    'Precision (Macro Avg)': precision,
    'F1-Score (Macro Avg)': f1
}
print("\n\033[1mClassification Metrics:\033[1m")
classification_metrics = pd.DataFrame(classification_metrics.items(), columns=['Metric', 'Value'])
classification_metrics['Value'] = classification_metrics['Value'].round(3)
display(classification_metrics)
 
 ```   


### `visualization` Module

####
 `vary_number_fewshot_example(...)`
Vary the size of the learning set and evaluate classification performance.

####
- `create_confusion_matrix_gif(...)`
Creates a GIF of confusion matrix images and displays it.

####
- `preprocess_and_plot_learning_size_metrics(...)`
Preprocesses raw repeated-block data and plots learning size metrics.

####
- `preprocess_and_plot_two_models(...)`
Preprocesses two raw datasets and plots learning size metrics for both.


## Additional Notes

**Dependencies:** Ensure all required dependencies are installed to avoid import errors.

**Testing:** For contributors, run the tests using pytest:

```bash
pytest tests/
```

Feedback and Contributions: Feel free to contribute by creating a pull request or reporting issues on the GitHub repository.



