from multiprocessing import process
import numpy as np
from dataset import get_multi_images, get_single_image, normalize_image
from llm.level_description import description_with_gpt
from superpixel import superpixel as sp
from yolo.yolo import get_mask_from_YOLO


def run_by_option(option: int) -> None:
    # < Generate a description from a file (one by one) >
    if option == "1":
        print("=> You selected option 1.\n")

        """
            [ Process the image file ]
        """
        # Load an image file
        image = get_single_image()
        # Normalize the image
        norm_image = normalize_image(image)

    # < Generate series of descriptions from a folder >
    else:
        print("=> You selected option 2.\n")

        """
            [ Process the image file ]
        """
        # Load images from a folder
        images = get_multi_images()
        # Normalize the images
        norm_images = []
        for image in images:
            norm_images.append(normalize_image(image))


def get_mask_option() -> bool:
    print("[Select the mask option]")
    print("1. Use Mask from YOLO")
    print("2. Do not use Mask")

    while True:
        try:
            mask_option = int(input("Enter the number: "))
            if mask_option in [1, 2]:
                break
            elif mask_option == 3:
                print("Exiting...")
                return mask_option
            else:
                print(
                    "Invalid option. Please enter 1, 2, or 3.\n1. Use Mask from YOLO\n2. Do not use Mask\n3. Exit")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return mask_option


def get_yolo_superpixel(image: np.ndarray, mask_true: bool = 1) -> np.ndarray:
    # Apply superpixel
    if mask_true:
        # Get the mask from YOLO
        yolo_model = "yolov8l-seg.pt"
        mask_dict = get_mask_from_YOLO(image, yolo_model=yolo_model)

        processed_img = sp.superpixel(image, mask_dict)
    else:
        processed_img = sp.superpixel_no_mask(image)

    return processed_img


def get_description(orig_image: np.ndarray, sup_image: np.ndarray, level_option: int) -> str | list[str]:
    # Level options
    levels = [1, 2, 3, 4]
    all = 5

    # Get the description one by one
    if level_option in levels:
        description = description_with_gpt(
            orig_image, sup_image).processing_image_with_level(level_option)
        return description

    # Get the description for all levels
    if level_option == all:
        desc_obj = description_with_gpt(orig_image, sup_image)
        descriptions = [desc_obj.processing_image_with_level(
            level) for level in levels]
        return descriptions


def save_description(description: str, file_path: str) -> None:
    with open(file_path, "w") as file:
        file.write(description)
        print(f"Description saved to {file_path}")


def ask_level_option() -> int:
    print("""[Select the description level]
1. Simple Text-Only
2. Simple Text + Superpixel Image
3. Structured Text-Only
4. Structured Text + Superpixel Image
5. Get All at Once
6. Close""")

    while True:
        try:
            level_option = int(input("Enter the number: "))
            if level_option in [1, 2, 3, 4, 5, 6]:
                return level_option
            else:
                print(
                    "Invalid option. Please enter 1 ~ 6.")
        except ValueError:
            print("Invalid input. Please enter a number.")
