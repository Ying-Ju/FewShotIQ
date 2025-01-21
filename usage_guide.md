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
Preprocesses one or two raw repeated-block datasets and plots learning size metrics.


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

In order to use the functions `evaluate_zero_shot_predictions()` and `few_shot_fault_classification` in this module, we need to import necessary libraries, set the OpenAI API key, saved folder location, and load images first. The following code gives an example for this purpose. In the example below, users should replace `OPENAI_API_KEY` by their own key and replace `/content` by their save folder. The dataset used in this example were from [Megahed and Camelio (2012)](https://doi.org/10.1007/s10845-010-0378-3). We utilized their learning dataset comprising 10 nominal and 10 defective images, alongside their testing dataset consisting of 50 nominal images and 50 defective images.


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
The following code chunk gives an example to compute standard classification metrics such as accuracy, precision, recall, and F1-score from a CSV file containing classification results. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 75 images for each defective class (Band, Bimodal, Single Crystal). Our example file `results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/usage_examples/results.csv).


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

Here, we use an example to show the use of these two functions: `map_multiclass_labels` and `calculate_specificity()` to compute major label accuracy for multiclass. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 75 images for each defective class (Band, Bimodal, Single Crystal). The example file `results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/usage_examples/results.csv).

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

Similar to the previous example, we show the use of these two functions: `map_multiclass_labels` and `calculate_specificity()` to compute per sub label accuracy for multiclass. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 25 images for each defective class (Low Band, Medium Band, High Band, Low Bimodal, Medium Bimodal, High Bimodal, Low Single, Medium Single, High Single). Our example file `results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/usage_examples/results.csv).


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

In this module, We will introduce several functions for visualzing classification performance via different situations. First, we import necessary libraries, set the OpenAI API key, saved folder location, and load images. The following code gives an example for this purpose. In the example below, users should replace `OPENAI_API_KEY` by their own key and replace `/content` by their save folder. The dataset used in this example were images of stochastic textured surfaces (STS), which can be found in our pakcage. 


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

# Set the GitHub repository details
repo_owner = "fmegahed"
repo_name = "qe_genai"
base_path = "data/textile_images/simulated"

# Retrieve URLs for each subfolder
nominal_image_urls = get_image_urls(repo_owner, repo_name, base_path, "nominal")
defective_local_image_urls = get_image_urls(repo_owner, repo_name, base_path, "local")
defective_global_image_urls = get_image_urls(repo_owner, repo_name, base_path, "global")

# Set seed for reproducibility
random.seed(2024)

# Step 1: Select Test Set
# Set test set size
test_size = 100
test_nominal_image_urls = random.sample(nominal_image_urls, test_size)
test_defective_global_image_urls = random.sample(defective_global_image_urls, int(test_size / 2))
test_defective_local_image_urls = random.sample(defective_local_image_urls, int(test_size / 2))

# Combine defective test set
test_defective_image_urls = test_defective_global_image_urls + test_defective_local_image_urls

# Step 2: Create Lists of Remaining Images
# Remove test images from original lists
remaining_nominal_image_urls = [url for url in nominal_image_urls if url not in test_nominal_image_urls]
remaining_defective_global_image_urls = [url for url in defective_global_image_urls if url not in test_defective_global_image_urls]
remaining_defective_local_image_urls = [url for url in defective_local_image_urls if url not in test_defective_local_image_urls]

# Extract test image filenames
test_nominal_image_filenames = [url.split('/')[-1] for url in test_nominal_image_urls]
test_defective_global_filenames = [url.split('/')[-1] for url in test_defective_global_image_urls]
test_defective_local_filenames = [url.split('/')[-1] for url in test_defective_local_image_urls]
test_image_filenames = test_nominal_image_filenames + test_defective_global_filenames + test_defective_local_filenames

# Step 3: Shuffle Lists
random.shuffle(remaining_nominal_image_urls)
random.shuffle(remaining_defective_global_image_urls)
random.shuffle(remaining_defective_local_image_urls)

# Load the testing images from GitHub
test_nominal_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in test_nominal_image_urls]
test_global_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in test_defective_global_image_urls]
test_local_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in test_defective_local_image_urls]
test_defective_images = test_global_images + test_local_images

test_images = (test_nominal_images + test_defective_images)

# Preloading the Images for Learning:

learn_nominal_image_filenames = [url.split('/')[-1] for url in remaining_nominal_image_urls]
learn_defective_global_filenames = [url.split('/')[-1] for url in remaining_defective_global_image_urls]
learn_defective_local_filenames = [url.split('/')[-1] for url in remaining_defective_local_image_urls]

# Descriptions for learning images
nominal_descriptions = [
    f"Image {filename}: An image of a textile material with consistent weave patterns, showing no visible defects or irregularities."
    for filename in learn_nominal_image_filenames
]
global_descriptions = [
    f"Image {filename}: A textile pattern with a slight overall distortion from our baseline textile, causing a uniform shift across the entire surface."
    for filename in learn_defective_global_filenames
]
local_descriptions = [
    f"Image {filename}: An image of a textile material with a small localized defect, disrupting the otherwise consistent weave pattern."
    for filename in learn_defective_local_filenames
]

defective_descriptions = global_descriptions + local_descriptions

# Loading the learning images from GitHub
nominal_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in remaining_nominal_image_urls]
global_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in remaining_defective_global_image_urls]
local_images = [preprocess(load_image(url)).unsqueeze(0).to(device) for url in remaining_defective_local_image_urls]

defective_images = global_images + local_images
```

#### Varying the Few Shot Learning Size

Suppose users have imported libraries, set the OpenAI API Key, saved folder location, and loaded images. Otherwise, please review the previous sections for the relevant topics. We give an example to vary the size of the learning set and evaluate classification performance using the function `vary_number_fewshot_examples()`, which saves classification metrics and confusion matrices to files. Note: It may takes a lot of time to obtain the results. 

```python
from FewShotIQ.classification import few_shot_fault_classification
from FewShotIQ.evaluation import compute_classification_metrics
from FewShotIQ.visualization import vary_number_fewshot_examples

# Running the Function:
# ---------------------
vary_number_fewshot_examples(
    learn_grid = [10, 20, 30, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 350],
    nominal_images = nominal_images,
    defect_dict={
    "global": global_images,
    "local": local_images
    },
    test_images= test_images,
    test_image_filenames=test_image_filenames,
    nominal_descriptions=nominal_descriptions,
    defect_descriptions={
    "global": global_descriptions,
    "local": local_descriptions
    },
    equal_learn_size=True,
    defect_learn_size=50,  # Ignored if equal_learn_size=True
    save_folder=save_folder,
    prefix="exp03",  # Prefix for saved CSV and PNG file names
    labels=["Nominal", "Defective"],  # Labels for classification
    label_counts=[len(test_nominal_images), len(test_defective_images)]  # Number of test samples per class
)
```

#### Visualizing the Confusion Matrices

Follow the previous example for applying `vary_number_fewshot_examples{}` function, we create a GIF on confusion matrix images using `create_confusion_matrix_gif()` and display it. This function reads images of confusion matrices after applying `vary_number_fewshot_examples{}` function. The argument *save_folder* in the function indicates where these images are saved. 


```python
import os
from PIL import Image
from IPython.display import Image as IPImage, display
import pandas as pd
import matplotlib.pyplot as plt
from FewShotIQ.visualization import create_confusion_matrix_gif

create_confusion_matrix_gif(
    prefix="exp03_conf_matrix",
    learn_size=[10, 20, 30, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 350],
    duration=3000,
    save_folder=save_folder,
    file_save_name= "exp03_conf_matrices_all.gif"
)
```

#### Visualizing the Impact of Learning Size on Classification Metrics

Follow the previous example for applying `vary_number_fewshot_examples{}` function, we preprocesses raw repeated-block data and plots learning size metrics. In the example, the *exp03_aggregated_results.csv* was an output from the `vary_number_fewshot_examples{}` function. One should note that "exp03" is the prefix for saved CSV and PNG file names in this example. Our example *exp03_aggregated_results.csv* can be found [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/usage_examples/exp03_aggregated_results.csv).

```python
import os
from PIL import Image
from IPython.display import Image as IPImage, display
import pandas as pd
import matplotlib.pyplot as plt
from FewShotIQ.visualization import preprocess_and_plot_learning_size_metrics

raw_results = pd.read_csv(f"{save_folder}/exp03_aggregated_results.csv")
preprocess_and_plot_learning_size_metrics(
    raw_data1=raw_results,
    save_folder=save_folder,
    save_file_name="exp03_learning_size_metrics.png",
    y_limits=[0, 1.07],
    label_positions={10, 50, 100, 150, 200, 250, 300, 350},
    suptitle="Impact of Learning Set Size on Classification Metrics for Experiment 03",
    figure_size=(15, 10)
)

```

#### Visualizing the Impact of Model Choice on Classification Metrics

The following code chunk shows an example to use the `preprocess_and_plot_learning_size_metrics()` function to preprocess two raw datasets, which are output files by applying `vary_number_fewshot_examples{}` function using two models individually and plots learning size metrices for both. In the example below, these two models are *ViT-L/14* and *vit-b/32*. One should note that "exp03" is the prefix for saved CSV and PNG file names in this example. The example files *exp03_aggregated_results.csv* can be found [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/usage_examples/exp03_aggregated_results.csv) and *exp03_b32_aggregated_results.csv* can be founded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/FewShotIQ/data/usage_examples/exp03_b32_aggregated_results.csv).


```python
import pandas as pd
import matplotlib.pyplot as plt
from FewShotIQ.visualization import preprocess_and_plot_learning_size_metrics

dataset1 = pd.read_csv(f"{save_folder}/exp03_aggregated_results.csv")
dataset2 = pd.read_csv(f"{save_folder}/exp03_b32_aggregated_results.csv")

preprocess_and_plot_learning_size_metrics(
    raw_data1=dataset1,
    save_folder=save_folder,
    save_file_name="exp03_comparing_two_clip_models.png",
    y_limits=[0, 1.07],
    label_positions={10, 50, 100, 150, 200, 250, 300, 350},
    suptitle="Learning Size Metrics Comparison",
    figure_size=(15, 10),
    raw_data2=dataset2,
    colors=['#1b9e77', '#d95f02'],
    labels=['ViT-L/14', 'ViT-B/32']
)
````

## Additional Notes

**Dependencies:** Ensure all required dependencies are installed to avoid import errors.

**Testing:** For contributors, run the tests using pytest:

```bash
pytest tests/
```

Feedback and Contributions: Feel free to contribute by creating a pull request or reporting issues on the GitHub repository.



