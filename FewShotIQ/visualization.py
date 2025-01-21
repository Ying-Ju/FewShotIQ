
from IPython.display import Image as IPImage, display
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from typing import List, Optional, Dict
from FewShotIQ.classification import few_shot_fault_classification
from FewShotIQ.evaluation import compute_classification_metrics


def vary_number_fewshot_examples(
    learn_grid: List[int],
    nominal_images: List[Image.Image],
    defect_dict: Dict[str, List[Image.Image]],
    test_images: List[Image.Image],
    test_image_filenames: List[str],
    nominal_descriptions: List[str],
    defect_descriptions: Dict[str, List[str]],
    equal_learn_size: bool = True,
    defect_learn_size: Optional[int] = None,
    save_folder: str = './results',
    prefix: str = "exp",
    labels: List[str] = ["Nominal", "Defective"],
    label_counts: List[int] = [50, 50]
) -> None:
    """
    Vary the size of the learning set and evaluate classification performance.

    Parameters:
    - learn_grid (List[int]): List of learning set sizes to evaluate.
    - nominal_images (List[Image.Image]): List of preprocessed nominal images.
    - defect_dict (Dict[str, List[Image.Image]]): Dictionary of defect types with associated images.
    - test_images (List[Image.Image]): List of preprocessed test images.
    - test_image_filenames (List[str]): Corresponding filenames for test images.
    - nominal_descriptions (List[str]): Descriptions for nominal images.
    - defect_descriptions (Dict[str, List[str]]): Dictionary of defect types with associated descriptions.
    - equal_learn_size (bool): Whether to enforce equal learning size for nominal and defective images. Default is True.
    - defect_learn_size (Optional[int]): Number of defective images used for learning if `equal_learn_size` is False.
    - save_folder (str): Path to save results. Default is './results'.
    - prefix (str): Prefix for saved CSV and PNG file names. Default is "exp".
    - labels (List[str]): List of labels for classification. Default is ["Nominal", "Defective"].
    - label_counts (List[int]): List of counts for each label in `labels`. Default is [50, 50].

    Returns:
    - None: Saves classification metrics and confusion matrices to files.
    """
    results = []

    for learn_size in learn_grid:
        # Determine the size for each defect type
        if equal_learn_size:
            defect_type_learn_size = learn_size // len(defect_dict)
        else:
            if defect_learn_size is None:
                raise ValueError("If `equal_learn_size` is False, `defect_learn_size` must be provided.")
            defect_type_learn_size = defect_learn_size // len(defect_dict)

        # Prepare nominal learning set
        nominal_learn_images = nominal_images[:learn_size]

        # Prepare defective learning set
        defective_learn_images = []
        defective_learn_descriptions = []
        for defect_type, images in defect_dict.items():
            defective_learn_images += images[:defect_type_learn_size]
            defective_learn_descriptions += defect_descriptions[defect_type][:defect_type_learn_size]

        # Perform classification
        csv_filename = f"{prefix}_results_{learn_size}.csv"

        few_shot_fault_classification(
            test_images=test_images,
            test_image_filenames=test_image_filenames,
            nominal_images=nominal_learn_images,
            nominal_descriptions=nominal_descriptions[:learn_size],
            defective_images=defective_learn_images,
            defective_descriptions=defective_learn_descriptions,
            num_few_shot_nominal_imgs=learn_size,
            file_path=save_folder,
            file_name=csv_filename
        )

        # Compute metrics
        metrics_df = compute_classification_metrics(
            csv_file=f"{save_folder}/{csv_filename}",
            labels=labels,
            label_counts=label_counts,
            num_few_shot_nominal_imgs=learn_size,
            save_confusion_matrix=True,
            cm_file_path=save_folder,
            cm_file_name=f"{prefix}_conf_matrix_{learn_size}.png"
        )

        results.append(metrics_df)

    # Save aggregated results
    aggregated_results = pd.concat(results, ignore_index=True)
    aggregated_results.to_csv(f"{save_folder}/{prefix}_aggregated_results.csv", index=False)



def create_confusion_matrix_gif(prefix, learn_size, duration, save_folder, file_save_name):
    """
    Creates a GIF of confusion matrix images and displays it.

    Args:
        prefix (str): Prefix for confusion matrix image files.
        learn_size (list): List of number of nominal learning images to iterate over.
        duration (int): Duration in milliseconds for each frame in the GIF.
        save_folder (str): Folder containing the confusion matrix images.
        file_save_name (str): Name of the saved GIF file.

    Returns:
        None
    """
    # Create a list of paths for the confusion matrix images
    image_paths = [
        os.path.join(save_folder, f"{prefix}_{size}.png")
        for size in learn_size
        if os.path.exists(os.path.join(save_folder, f"{prefix}_{size}.png"))
    ]

    # Load the images
    images = [Image.open(image_path) for image_path in image_paths]

    # Save as GIF
    gif_path = os.path.join(save_folder, file_save_name)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)

    # Display the GIF
    display(IPImage(gif_path))


def preprocess_and_plot_learning_size_metrics(
    raw_data1, 
    save_folder, 
    save_file_name, 
    y_limits, 
    label_positions, 
    suptitle, 
    figure_size, 
    raw_data2=None, 
    colors=None, 
    labels=None
):
    """
    Preprocesses raw data and plots learning size metrics. Handles one or two datasets.

    Args:
        raw_data1 (pd.DataFrame): First raw DataFrame with repeated blocks for each learning size.
        save_folder (str): Folder to save the plot image.
        save_file_name (str): Name of the saved plot file.
        y_limits (list): Y-axis limits for all plots [y_min, y_max].
        label_positions (set): X-values where labels and black markers should be displayed.
        suptitle (str): The main title for the plot.
        figure_size (tuple): Size of the figure (width, height).
        raw_data2 (pd.DataFrame, optional): Second raw DataFrame for comparison. Defaults to None.
        colors (list, optional): List of colors for datasets (e.g., ['blue', 'orange']). Defaults to None.
        labels (list, optional): List of labels for datasets (e.g., ['Dataset 1', 'Dataset 2']). Defaults to None.

    Returns:
        None
    """
    def preprocess(raw_data):
        """Preprocesses a single dataset."""
        blocks = []
        num_few_shot_nominal_imgs = None
        for index, row in raw_data.iterrows():
            if "Nominal Learning Size" in row["Metric"]:
                num_few_shot_nominal_imgs = float(row["Value"])
            else:
                blocks.append({
                    "Nominal Learning Size": num_few_shot_nominal_imgs,
                    row["Metric"]: float(row["Value"])
                })
        return pd.DataFrame(blocks).groupby("Nominal Learning Size").first().reset_index()

    # Preprocess the data
    processed_data1 = preprocess(raw_data1)
    processed_data2 = preprocess(raw_data2) if raw_data2 is not None else None

    # Define metrics and their display names
    metrics = ["Accuracy", "Specificity", "Sensitivity (Recall)", "Precision", "F1 Score", "AUC"]
    metric_titles = ["Accuracy", "Specificity", "Sensitivity", "Precision", "F1 Score", "AUC"]

    # Plot the data
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figure_size)
    axes = axes.flatten()

    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i]

        # Plot first dataset
        ax.plot(
            processed_data1["Nominal Learning Size"],
            processed_data1[metric],
            marker='o',
            color=colors[0] if colors else 'grey',
            linewidth=1.5,
            markersize=5,
            label=labels[0] if labels else 'Dataset 1'
        )

        # Highlight labels for first dataset
        for x, y in zip(processed_data1["Nominal Learning Size"], processed_data1[metric]):
            if x in label_positions:
                ax.plot(x, y, marker='o', color='black', markersize=6)
                ax.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=10, color='black')

        # Plot second dataset if provided
        if processed_data2 is not None:
            ax.plot(
                processed_data2["Nominal Learning Size"],
                processed_data2[metric],
                marker='s',
                color=colors[1] if colors else 'orange',
                linewidth=1.5,
                markersize=5,
                label=labels[1] if labels else 'Dataset 2'
            )

            # Highlight labels for second dataset
            for x, y in zip(processed_data2["Nominal Learning Size"], processed_data2[metric]):
                if x in label_positions:
                    ax.plot(x, y, marker='s', color='black', markersize=6)
                    ax.text(x, y + 0.02, f"{y:.2f}", ha='center', va='bottom', fontsize=10, color='black')

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Nominal Learning Set Size", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_ylim(y_limits)
        ax.grid(False)
        if labels:
            ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, hspace=0.3)
    plt.suptitle(suptitle, fontsize=16)
    plt.figtext(
        0.5, -0.05,
        "Note: Black markers at specified points highlight values for clarity, "
        "showing the trend in metric values as learning size increases.",
        ha="center", fontsize=10
    )
    save_path = f"{save_folder}/{save_file_name}"
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()