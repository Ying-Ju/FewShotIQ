import importlib.resources as pkg_resources
from io import BytesIO
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os
from pathlib import Path
from PIL import Image
import requests
import textwrap
from typing import List, Optional, Union
from urllib.parse import urlparse

import unittest
from unittest.mock import MagicMock, patch
from FewShotIQ.image_utils import get_image_urls, load_package_image, load_image, animate_images

class TestImageUtils(unittest.TestCase):

    @patch("FewShotIQ.image_utils.requests.get")  # Mock the requests.get method
    def test_get_image_urls(self, mock_get):
        """Test get_image_urls function."""

        # Arrange: Mock GitHub API response
        repo_owner = "fmegahed"
        repo_name = "qe_genai"
        base_path = "data/pan_images/train"
        subfolder = "nominal"

        mock_response_data = [
            {'name': '1.JPG', 
            'path': 'data/pan_images/train/nominal/1.JPG', 
            'sha': '4ef8a89fde95e83ae569753c5a93c74225126487', 
            'size': 1505468, 
            'url': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/1.JPG?ref=main',
            'html_url': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/1.JPG', 
            'git_url': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/4ef8a89fde95e83ae569753c5a93c74225126487', 
            'download_url': 'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/1.JPG', 
            'type': 'file', 
            '_links': {'self': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/1.JPG?ref=main', 
            'git': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/4ef8a89fde95e83ae569753c5a93c74225126487', 
            'html': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/1.JPG'}
            }, 
            {'name': '11.JPG',
            'path': 'data/pan_images/train/nominal/11.JPG', 
            'sha': '9a24648a57798a2a275b2b7a6fda967c00b83d6d', 
            'size': 1520341, 
            'url': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/11.JPG?ref=main', 
            'html_url': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/11.JPG', 
            'git_url': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/9a24648a57798a2a275b2b7a6fda967c00b83d6d', 
            'download_url': 'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/11.JPG', 
            'type': 'file', 
            '_links': {'self': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/11.JPG?ref=main', 
            'git': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/9a24648a57798a2a275b2b7a6fda967c00b83d6d', 
            'html': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/11.JPG'}
            }
        ]

        mock_get.return_value.json.return_value = mock_response_data
        mock_get.return_value.status_code = 200

        # Act: Call the function
        result = get_image_urls(repo_owner, repo_name, base_path, subfolder)

        # Assert: Verify the results
        expected = [
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/1.JPG',
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/11.JPG'
        ]
        self.assertEqual(result, expected)

    @patch("FewShotIQ.image_utils.requests.get")
    def test_get_image_urls_no_required_images(self, mock_get):
        """Test get_image_urls when no matching images are found."""

        # Mock the API response with non-image files
        mock_get.return_value.json.return_value = [
            {'name': '1.JPG', 
            'path': 'data/pan_images/train/nominal/1.JPG', 
            'sha': '4ef8a89fde95e83ae569753c5a93c74225126487', 
            'size': 1505468, 
            'url': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/1.JPG?ref=main',
            'html_url': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/1.JPG', 
            'git_url': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/4ef8a89fde95e83ae569753c5a93c74225126487', 
            'download_url': 'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/1.JPG', 
            'type': 'file', 
            '_links': {'self': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/1.JPG?ref=main', 
            'git': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/4ef8a89fde95e83ae569753c5a93c74225126487', 
            'html': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/1.JPG'}
            }, 
            {'name': '11.JPG',
            'path': 'data/pan_images/train/nominal/11.JPG', 
            'sha': '9a24648a57798a2a275b2b7a6fda967c00b83d6d', 
            'size': 1520341, 
            'url': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/11.JPG?ref=main', 
            'html_url': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/11.JPG', 
            'git_url': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/9a24648a57798a2a275b2b7a6fda967c00b83d6d', 
            'download_url': 'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/11.JPG', 
            'type': 'file', 
            '_links': {'self': 'https://api.github.com/repos/fmegahed/qe_genai/contents/data/pan_images/train/nominal/11.JPG?ref=main', 
            'git': 'https://api.github.com/repos/fmegahed/qe_genai/git/blobs/9a24648a57798a2a275b2b7a6fda967c00b83d6d', 
            'html': 'https://github.com/fmegahed/qe_genai/blob/main/data/pan_images/train/nominal/11.JPG'}
            }
        ]
        mock_get.return_value.status_code = 200

        # Inputs
        repo_owner = "fmegahed"
        repo_name = "qe_genai"
        base_path = "data/pan_images/train"
        subfolder = "nominal"
        file_extensions = [".pdf"]

        # Call the function
        result = get_image_urls(repo_owner, repo_name, base_path, subfolder, "main", file_extensions)

        # Assert: Should return an empty list since we want to get only PDF files
        self.assertEqual(result, [])

    @patch("FewShotIQ.image_utils.requests.get")
    def test_get_image_urls_http_error(self, mock_get):
        """Test get_image_urls when an HTTP error occurs."""

        # Mock the API response to raise an HTTP error
        mock_get.return_value.status_code = 404

        # Inputs
        repo_owner = "fmegahed"
        repo_name = "qe_genai"
        base_path = "data/pan_images/train"
        subfolder = "nominal"
        

        # Assert: The function should handle the exception gracefully
        with self.assertRaises(Exception):
            get_image_urls(repo_owner, repo_name, base_path, subfolder, "main", file_extensions)

    
    def test_load_valid_package_image(self):
        # Arrange: Ensure the image exists in the package's data folder
        valid_image_path = "pan_images/test/defective/IMG_1514.JPG"  # Adjust this to your real file path

        img = load_package_image(valid_image_path)

        # Assert
        self.assertIsInstance(img, Image.Image)  # Check the return type is a PIL Image
        self.assertEqual(img.mode, "RGB")  # Verify the default mode


    @patch("FewShotIQ.image_utils.requests.get")
    def test_load_image(self, mock_get):
        """Test load_images with a mock URL."""
        
        # Mock the response for an image url
        with open("tests/resources/NewRiverGorge.jpg", "rb") as f:
            real_image_data = f.read()
        mock_get.return_value.content = real_image_data
        mock_get.return_value.status_code = 200
        
        # Call the function
        image = load_image('https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/1.JPG')

        # Verify the results
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.mode, "RGB")
        mock_get.return_value.close.assert_called_once()
        
    @patch("FewShotIQ.image_utils.plt.show")
    @patch("FewShotIQ.image_utils.FuncAnimation")
    def test_animate_images(self, mock_animation, mock_show):
        # Mock image information
        nominal_image_urls = ['https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/1.JPG',
                              'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/11.JPG',
                              'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/6.JPG']
        nominal_labels = ['nominal', 'nominal', 'nominal']
        defective_image_urls = ['https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/defective/IMG_1572.JPG',
                                'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/defective/IMG_1573.JPG',
                                'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/defective/IMG_1574.JPG']
        defective_labels = ['defective', 'defective', 'defective']
        
        mock_animation.return_value = FuncAnimation(MagicMock(), lambda x: None)
        
        # Call the function
        animation = animate_images(nominal_image_urls, 
                                   nominal_labels, 
                                   defective_image_urls, 
                                   defective_labels,
                                   save_fig=False)
   
        # Assert: Verify the result
        self.assertIsInstance(animation, FuncAnimation)  
        mock_show.assert_called_once() # Ensure plt.show() was called  

    
    @patch("FewShotIQ.image_utils.FuncAnimation")
    @patch("FewShotIQ.image_utils.plt.close")
    def test_animate_images_save(self, mock_close, mock_animation):
        """Test animate_images with file saving."""
        # Arrange
        nominal_image_urls = [
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/1.JPG',
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/11.JPG',
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/nominal/6.JPG'
        ]
        nominal_labels = ['nominal', 'nominal', 'nominal']
        defective_image_urls = [
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/defective/IMG_1572.JPG',
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/defective/IMG_1573.JPG',
            'https://raw.githubusercontent.com/fmegahed/qe_genai/main/data/pan_images/train/defective/IMG_1574.JPG'
        ]
        defective_labels = ['defective', 'defective', 'defective']

        mock_animation.return_value = MagicMock(spec=FuncAnimation)

        file_path = "tests/resources"
        file_name = "test_animation.gif"

        # Act
        animation = animate_images(
            nominal_image_urls,
            nominal_labels,
            defective_image_urls,
            defective_labels,
            save_fig=True,
            file_path=file_path,
            file_name=file_name,
        )

        # Assert
        mock_animation.return_value.save.assert_called_once_with(f"{file_path}/{file_name}", writer="pillow")
        mock_close.assert_called_once()
        assert isinstance(animation, FuncAnimation)



if __name__ == "__main__":
    unittest.main()
