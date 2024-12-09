# Import submodules to expose their functionality
from .image_utils import get_image_urls, load_package_image, load_image, animate_images



# Define what gets exposed when importing the package
__all__ = [
    "get_image_urls", 
    "load_image", 
    "animate_images",
    "evaluate_zero_shot_predictions", 
    "load_image", 
    "animate_images",
    "compute_classification_metrics",
    "vary_number_fewshot_examples", 
    "create_confusion_matrix_gif", 
    "preprocess_and_plot_learning_size_metrics", 
    "preprocess_and_plot_two_models",
]

# Version of the package
__version__ = "0.1.0"