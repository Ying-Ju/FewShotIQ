# Import submodules to expose their functionality
from .image_utils import get_image_urls, load_package_image, load_image, animate_images
from .classification import evaluate_zero_shot_predictions, few_shot_fault_classification
from .evaluation import compute_classification_metrics, map_multiclass_labels, calculate_specificity
from .visualization import (
    vary_number_fewshot_examples, 
    create_confusion_matrix_gif, 
    preprocess_and_plot_learning_size_metrics
)


# Define what gets exposed when importing the package
__all__ = [
    "get_image_urls",
    "load_package_image", 
    "load_image", 
    "animate_images",
    "evaluate_zero_shot_predictions", 
    "few_shot_fault_classification",
    "compute_classification_metrics",
    "map_multiclass_labels",
    "calculate_specificity",
    "vary_number_fewshot_examples", 
    "create_confusion_matrix_gif", 
    "preprocess_and_plot_learning_size_metrics", 
]

# Version of the package
__version__ = "0.1.0"