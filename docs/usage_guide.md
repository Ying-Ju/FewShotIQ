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

#### Loading Images 

The following code chunk gives examples to load an image from a local path or a URL using the `load_image()` function.

```python
from urllib.parse import urlparse  # To check if the input is a URL
from io import BytesIO            # To handle binary data from the URL
import requests                   # To fetch the image from a URL
from PIL import Image             # To load and process images
from FewShotIQ.image_utils import load_image

# Load an image from a local path
img = load_image("tests/resources/NewRiverGorge.jpg")
img.show()

# Load an image from a URL
img = load_image("https://github.com/Ying-Ju/FewShotIQ/blob/main/usage_examples/Love_River_Taiwan.jpg?raw=true")
img.show()
```

#### Create a GIF of animated images

The following code chunk gives an example using the `animate_images()` function to animate nominal and defective images side-by-side with resizing only for images larger than 800x800 pixels. We use the image data from the GitHub Repository [qe_genai](https://github.com/fmegahed/qe_genai/tree/main/). This example also shows how to retrieve the URLs of files in a specified GitHub repository subfolder using the `get_image_urls()` function. Users must change the *save_folder* in the function input of `animate_images()` to the directory where they want to store the GIF file. 


```python
from urllib.parse import urlparse # To determine if a string is a URL
from io import BytesIO            # To handle binary image data from the URL
import requests                   # To fetch images from URLs
from PIL import Image             # To load and process images
import textwrap                   # To wrap text for subtitles
import matplotlib.pyplot as plt   # To create plots
from matplotlib.animation import FuncAnimation  # To create animations
from typing import List, Optional, Dict, Any # For type annotations
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
    file_path=save_folder,  # replace save_folder with a directory where the GIF file will be saved
    file_name="pan_learning_images.gif"
)
```

### `classification` Module

In order to use the functions `evaluate_zero_shot_predictions()` and `few_shot_fault_classification` in this module, we need to import necessary libraries, set the OpenAI API key, saved folder location, and load images first. The following code gives an example for this purpose. In the example below, users should replace `OPENAI_API_KEY` by their own key and replace `/content` by their save folder. The dataset used in this example were from [Megahed and Camelio (2012)](https://doi.org/10.1007/s10845-010-0378-3). We utilized their learning dataset comprising 10 nominal and 10 defective images, alongside their testing dataset consisting of 50 nominal images and 50 defective images.


```python
import os                           # To work with file paths and directories
import random                       # To shuffle or select random items
import importlib.util
import subprocess

# packages that need to be installed in the colab
packages = ['openai', 'torch', 'pillow', 'requests', 'clip', 'typing']

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

from typing import List, Optional, Dict, Any    # For type annotations
import openai                                   # To use the OpenAI API
import clip                                     # For CLIP model loading and preprocessing
import torch                                    # For tensor operations and checking CUDA availability
from PIL import Image                           # To handle image loading and manipulation
import torch.nn.functional as F

import random
from FewShotIQ.image_utils import get_image_urls, load_image

# package for importing the hidden key
openai.api_key = 'OPENAI_API_KEY' # insert your key here

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

from IPython.display import Image as IPyImage, display, clear_output
from FewShotIQ.classification import evaluate_zero_shot_predictions

zero_shot_metrics, zero_shot_df = evaluate_zero_shot_predictions(
    labels=['A metallic pan free of black scuff marks', 'A metallic pan with a simulated scuff mark drawn by a black marker'],
    label_counts= [50,50],
    test_images=test_images,
    test_image_filenames = [url.split('/')[-1] for url in test_image_urls],
    model=model,
    device=device,
    save_confusion_matrix = True,
    cm_title = "Confusion Matrix for Zero Shot Classification",
    short_labels=['Nominal', 'Defective'],
    cm_file_path = save_folder,  #save_folder should be replaced by the name of user's folder
    cm_file_name = "zero_shot_confusion_matrix.png"
)

print("\n\033[1mConfusion Matrix\033[1m\n")
display(zero_shot_metrics.round(3).set_index("Metric").T)

```

#### Few-Shot Classification with CLIP

Suppose users have imported libraries, set the OpenAI API Key, saved folder location, and loaded images. In that case, they can directly execute the `few_shot_fault_classification()` function for few-shot classification with CLIP. Otherwise, please review the previous sections for the relevant topics. 

```python
import os  # For file path and directory operations
import csv  # For writing classification results to a CSV file               
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
    file_name = 'pan_results.csv',   
    print_one_liner = False
)
```

### `evaluation` Module

### Computing Classification Metrics 
The following code chunk gives an example to compute standard classification metrics such as accuracy, precision, recall, and F1-score from a CSV file containing classification results. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `micro_results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 75 images for each defective class (Band, Bimodal, Single Crystal). Our example file `micro_results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/usage_examples/micro_results.csv).


```python
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
from IPython.display import Image as IPyImage, display, clear_output
from FewShotIQ.evaluation import compute_classification_metrics

classification_metrics = compute_classification_metrics(
    csv_file = f'{save_folder}/micro_results.csv',
    labels = ["Nominal", "Defective"],
    label_counts = [225, 225],
    num_few_shot_nominal_imgs = 225,
    save_confusion_matrix = True,
    cm_file_path = save_folder,
    cm_file_name = "micro_confusion_matrix.png"
    )

# Rounding for display
classification_metrics.Value = classification_metrics.Value.round(3)

# Drop the row corresponding to num_few_shot_nominal_imgs since it is not a classification metric
classification_metrics = classification_metrics.drop(index=0).reset_index(drop=True)

# Display the created confusion metrics figure
print("\n\033[1mConfusion Matrix\033[1m")
display(IPyImage(filename=f"{save_folder}/micro_confusion_matrix.png"))

# Display the metrics as a table with three digit precision
print("\n\033[1mClassification Metrics:\033[1m")
display(classification_metrics)
```


#### Computing Major Label Accuracy for Multiclass

Here, we use an example to show the use of these two functions: `map_multiclass_labels` and `calculate_specificity()` to compute major label accuracy for multiclass. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `micro_results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 75 images for each defective class (Band, Bimodal, Single Crystal) for the automated classification of microstructural material properties across parts. The example file `micro_results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/usage_examples/micro_results.csv).

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
from IPython.display import Image as IPyImage, display, clear_output
from FewShotIQ.evaluation import map_multiclass_labels, calculate_specificity

classification_results = pd.read_csv(f'{save_folder}/micro_results.csv')

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
plt.title("Confusion Matrix for Primary Multi-Class Labels")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(f"{save_folder}/micro_confusion_matrix_major_labels.png")
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

Similar to the previous example, we show the use of these two functions: `map_multiclass_labels` and `calculate_specificity()` to compute per sub label accuracy for multiclass. Suppose the `few_shot_fault_classification()` function was used to conduct few-shot classification with CLIP, and the result file `micro_results.csv` was saved in the `save_folder`, the folder you decided earlier. Our example includes 225 nominal images and 25 images for each defective class (Low Band, Medium Band, High Band, Low Bimodal, Medium Bimodal, High Bimodal, Low Single, Medium Single, High Single) for the automated classification of microstructural material properties across parts. Our example file `micro_results.csv` can be downloaded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/usage_examples/micro_results.csv).


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
from IPython.display import Image as IPyImage, display, clear_output
from FewShotIQ.evaluation import map_multiclass_labels, calculate_specificity

classification_results = pd.read_csv(f'{save_folder}/micro_results.csv')

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
plt.title("Confusion Matrix for Secondary Multi-Class Labels")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(f"{save_folder}/micro_confusion_matrix_minor_labels.png")
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
import os                           # To work with file paths and directories
import random                       # To shuffle or select random items
import importlib.util
import subprocess

# packages that need to be installed in the colab
packages = ['openai', 'torch', 'pillow', 'requests', 'clip', 'typing']

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

from typing import List, Optional, Dict, Any        # For type annotations
import openai                                       # To use the OpenAI API
import clip                                         # For CLIP model loading and preprocessing
import torch                                        # For tensor operations and checking CUDA availability
from PIL import Image                               # To handle image loading and manipulation
import torch.nn.functional as F

import random
from FewShotIQ.image_utils import get_image_urls, load_image

# package for importing the hidden key
openai.api_key = 'OPENAI_API_KEY' # insert your key here

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
    prefix="STS",  # Prefix for saved CSV and PNG file names
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
    prefix="STS_conf_matrix",
    learn_size=[10, 20, 30, 50, 60, 70, 80, 90, 100, 120, 150, 200, 250, 300, 350],
    duration=3000,
    save_folder=save_folder,
    file_save_name= "STS_conf_matrices_all.gif"
)
```

#### Visualizing the Impact of Learning Size on Classification Metrics

Follow the previous example for applying `vary_number_fewshot_examples{}` function, we preprocesses raw repeated-block data and plots learning size metrics. In the example, the *STS_aggregated_results.csv* was an output from the `vary_number_fewshot_examples{}` function. One should note that "STS" is the prefix for saved CSV and PNG file names in this example. Our example *STS_aggregated_results.csv* can be found [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/usage_examples/STS_aggregated_results.csv).

```python
import os
from PIL import Image
from IPython.display import Image as IPImage, display
import pandas as pd
import matplotlib.pyplot as plt
from FewShotIQ.visualization import preprocess_and_plot_learning_size_metrics

raw_results = pd.read_csv(f"{save_folder}/STS_aggregated_results.csv")
preprocess_and_plot_learning_size_metrics(
    raw_data1=raw_results,
    save_folder=save_folder,
    save_file_name="STS_learning_size_metrics.png",
    y_limits=[0, 1.07],
    label_positions={10, 50, 100, 150, 200, 250, 300, 350},
    suptitle="Impact of Learning Set Size on Classification Metrics",
    figure_size=(15, 10)
)

```

#### Visualizing the Impact of Model Choice on Classification Metrics

The following code chunk shows an example to use the `preprocess_and_plot_learning_size_metrics()` function to preprocess two raw datasets, which are output files by applying `vary_number_fewshot_examples{}` function using two models individually and plots learning size metrices for both. In the example below, these two models are *ViT-L/14* and *vit-b/32*. One should note that "STS" is the prefix for saved CSV and PNG file names in this example. The example files *STS_aggregated_results.csv* can be found [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/usage_examples/STS_aggregated_results.csv) and *STS_b32_aggregated_results.csv* can be founded [here](https://raw.githubusercontent.com/Ying-Ju/FewShotIQ/refs/heads/main/usage_examples/STS_b32_aggregated_results.csv).


```python
import os
from PIL import Image
from IPython.display import Image as IPImage, display
import pandas as pd
import matplotlib.pyplot as plt
from FewShotIQ.visualization import preprocess_and_plot_learning_size_metrics

dataset1 = pd.read_csv(f"{save_folder}/STS_aggregated_results.csv")
dataset2 = pd.read_csv(f"{save_folder}/STS_b32_aggregated_results.csv")

preprocess_and_plot_learning_size_metrics(
    raw_data1=dataset1,
    save_folder=save_folder,
    save_file_name="STS_comparing_two_clip_models.png",
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



