import cv2
import os
import numpy as np


def load_single_image(image_path: str) -> np.ndarray:
    """Loads an individual image file and returns it as a numpy array.

    Args:
        image_path (str): Path to the image file.

    Raises:
        FileNotFoundError: If the image file is not found.

    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image


def load_images_from_folder(folder_path: str = "data/using") -> list[np.ndarray]:
    """Loads all images from a specified folder and returns them as a list of numpy arrays.

    Args:
        folder_path (str, optional): Path to the folder containing images. Defaults to "data/using".

    Raises:
        ValueError: If no images are found in the folder.

    Returns:
        list: List of loaded images as numpy arrays.
    """
    images = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is not None:
            images.append(image)

    if not images:
        raise ValueError(f"No images found in folder: {folder_path}")

    return images


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes the image such that the longer side becomes 720px, maintaining aspect ratio.
    It interpolates the image using linear interpolation.

    Args:
        image (np.ndarray): The image to be normalized.

    Returns:
        np.ndarray: The normalized image.
    """
    height, width = image.shape[:2]

    if height > width:
        scale = 720 / height
    else:
        scale = 720 / width

    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image
