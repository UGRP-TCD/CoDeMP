import numpy as np
from dataset import get_multi_images, get_single_image, normalize_image
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


def do_yolo_and_superpixel(image: np.ndarray) -> dict:
    # Get the mask from YOLO
    yolo_model = "yolov8l-seg.pt"
    mask_dict = get_mask_from_YOLO(image, yolo_model=yolo_model)

    # Apply superpixel
    # processed_img = do superpixel()

    return processed_img
