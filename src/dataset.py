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


def get_single_image() -> np.ndarray:
    """Prompts the user to input the name of an image file to load and returns the loaded image.

    Returns:
        np.ndarray: The loaded image as a numpy array.
    """
    while True:
        file_name = input("Input the file name (<fileName>.<format>): ")

        try:
            image = load_single_image("data/using/" + file_name)
            print("Image loaded successfully!")
            return image
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("  a. Retry with a new file name")
            print("  b. Exit the program")

            while True:  # Invalid choice handling loop
                user_choice = input(
                    "Please select one of the following options (a / b): ").strip().lower()
                if user_choice == "b":
                    print("Exiting program...")
                    return
                elif user_choice == "a":
                    print("Retrying...")
                    break  # Exit the invalid choice loop and retry file input
                else:
                    print("Invalid choice. Please select 'a' or 'b'.")


def get_multi_images() -> list:
    """Prompts the user to input the path of a folder containing images to load and returns the loaded images.

    Returns:
        list: List of loaded images as numpy arrays.
    """
    while True:
        folder_path = input(
            "Input the folder path to load images(default - \"data/using\"): ").strip()

        try:
            images = load_images_from_folder(folder_path)
            print("Images loaded successfully!")
            return images
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("  a. Retry with a new folder path")
            print("  b. Exit the program")

            while True:  # Invalid choice handling loop
                user_choice = input(
                    "Please select one of the following options (a / b): ").strip().lower()
                if user_choice == "b":
                    print("Exiting program...")
                    return
                elif user_choice == "a":
                    print("Retrying...")
                    break  # Exit the invalid choice loop and retry folder input
                else:
                    print("Invalid choice. Please select 'a' or 'b'.")
