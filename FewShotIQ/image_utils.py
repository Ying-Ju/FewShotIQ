import importlib.resources as pkg_resources
from io import BytesIO
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from PIL import Image
import requests
import textwrap
from typing import List, Optional, Union
from urllib.parse import urlparse


def get_image_urls(
    repo_owner: str,
    repo_name: str,
    base_path: str,
    subfolder: str,
    branch: str = "main",
    file_extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Retrieve the URLs of files in a specified GitHub repository subfolder.

    Parameters:
    - repo_owner (str): Owner of the GitHub repository.
    - repo_name (str): Name of the GitHub repository.
    - base_path (str): Path within the repository to the folder containing subfolder(s).
    - subfolder (str): Specific subfolder to retrieve file URLs from.
    - branch (str, optional): Branch to pull files from. Default is 'main'.
    - file_extensions (Optional[List[str]], optional): List of file extensions to filter by (e.g., ["jpg", "png"]).
      If None, returns all files.

    Returns:
    - List[str]: List of raw GitHub URLs to the files within the specified subfolder.
    """
    # Construct the API URL
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{base_path}/{subfolder}?ref={branch}"

    # Attempt to fetch data from the GitHub API
    try:
        response = requests.get(api_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve files for {subfolder}. Error: {e}")
        return []

    # Filter and collect raw file URLs
    image_urls = []
    for file_info in response.json():
        if file_info['type'] == 'file':
            file_name = file_info['name']

            # Check if the file extension matches if specified
            if file_extensions is None or any(file_name.lower().endswith(ext.lower()) for ext in file_extensions):
                raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{base_path}/{subfolder}/{file_name}"
                image_urls.append(raw_url)

    return image_urls


def load_image(image_path_or_url: str, mode: str = "RGB") -> Image.Image:
    """
    Load an image from a local file path or a URL.

    Parameters:
    - image_path_or_url (str): The local file path or URL of the image.
    - mode (str, optional): Mode to convert the image to (e.g., "RGB", "L" for grayscale).
      Default is "RGB".

    Returns:
    - Image.Image: The loaded PIL image object in the specified mode.

    Raises:
    - ValueError: If the input is not a valid URL or file path.
    - IOError: If the image cannot be loaded from the given path or URL.
    """
    if not isinstance(image_path_or_url, str):
        raise ValueError("image_path_or_url must be a string representing a local file path or URL.")

    # Check if the input is a URL or a local file path
    if urlparse(image_path_or_url).scheme in ('http', 'https'):
        # Load the image from a URL
        try:
            response = requests.get(image_path_or_url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert(mode)
        except requests.exceptions.RequestException as e:
            raise IOError(f"Failed to load image from URL: {e}")
        finally:
            response.close()
    else:
        # Load the image from a local file path
        try:
            img = Image.open(image_path_or_url).convert(mode)
        except (FileNotFoundError, IOError) as e:
            raise IOError(f"Failed to load image from path: {e}")

    return img

def animate_images(
    nominal_image_urls: List[str],
    nominal_labels: List[str],
    defective_image_urls: List[str],
    defective_labels: List[str],
    resize_factor: float = 0.15,
    pause_time: float = 1.0,
    save_fig: bool = False,
    file_path: Optional[str] = None,
    file_name: str = "animation.gif"
):
    """
    Animate nominal and defective images side-by-side with resizing only for images larger than 800x800 pixels.

    Parameters:
    - nominal_image_urls (List[str]): List of URLs or paths for nominal images.
    - nominal_labels (List[str]): List of labels/descriptions for nominal images.
    - defective_image_urls (List[str]): List of URLs or paths for defective images.
    - defective_labels (List[str]): List of labels/descriptions for defective images.
    - resize_factor (float): Factor by which to resize the images (applied only to large images).
    - pause_time (float): Time in seconds between frames.
    - save_fig (bool): Whether to save the animation as a GIF.
    - file_path (Optional[str]): Path to save the GIF if save_fig is True.
    - file_name (str): Filename for the saved GIF if save_fig is True.

    Returns:
    - None
    """

    # Helper function to load and resize images
    def load_image_resized(url, resize_factor=0.15):
        """Load an image from a URL or local path and resize it only if it's larger than 800x800 pixels."""
        img = Image.open(BytesIO(requests.get(url).content) if url.startswith("http") else url)
        if img.width > 800 or img.height > 800:
            img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)))
        return img

    # Wrap text to specified width for subtitles
    def wrap_text(text, width=50):
        """Wrap text to the specified width."""
        return "\n".join(textwrap.wrap(text, width=width))

    # Set the number of frames based on the minimum length of nominal and defective images
    num_images = min(len(nominal_image_urls), len(defective_image_urls))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Adjust layout to give more space for titles
    fig.subplots_adjust(top=0.8)  # Increase top margin

    def update(frame):
        # Clear the previous frame
        axes[0].clear()
        axes[1].clear()

        # Load and resize images for the current frame
        nominal_img = load_image_resized(nominal_image_urls[frame], resize_factor)
        defective_img = load_image_resized(defective_image_urls[frame], resize_factor)

        # Get and wrap labels for the current frame
        nominal_label = wrap_text(nominal_labels[frame], width=50)
        defective_label = wrap_text(defective_labels[frame], width=50)

        # Display images with wrapped titles
        axes[0].imshow(nominal_img)
        axes[0].set_title(f"Nominal Image\n{nominal_label}", fontsize=10)  # Reduced font size
        axes[0].axis('off')

        axes[1].imshow(defective_img)
        axes[1].set_title(f"Defective Image\n{defective_label}", fontsize=10)  # Reduced font size
        axes[1].axis('off')

    # Create the animation
    ani = FuncAnimation(fig, update, frames=num_images, interval=pause_time * 1000)
    print(f"save_fig: {save_fig}, file_path: {file_path}, file_name: {file_name}")

    # Save or display the animation
    if save_fig and file_path:
        ani.save(f"{file_path}/{file_name}", writer="pillow")    
    else:
        plt.show()

    plt.close()
    return ani