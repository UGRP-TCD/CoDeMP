import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


def get_mask_from_YOLO(img: np.ndarray, yolo_model: str = "yolov8l-seg.pt") -> dict:
    """Get the mask from YOLO.

    Args:
        img (np.ndarray): The image to be processed.
        yolo_model (str, optional): The YOLO model to be used. Defaults to "yolov8l-seg.pt". See the documentation(https://docs.ultralytics.com/tasks/segment/#val) for more options.

    Returns:
        dict: Dictionary containing class names as keys and masks as values.
    """
    model = YOLO("pt/" + yolo_model)
    results = model(img)

    # Resize each mask to match resized_image's dimensions without extra padding
    if results[0].masks is not None:
        class_mask_dict = {}

        # Iterate through each mask
        for i, mask in enumerate(results[0].masks.data):
            # Convert mask to numpy array and resize to match resized_image's dimensions
            mask_np = mask.cpu().numpy()
            print(f"Original mask {i} dimensions: {mask_np.shape}")

            # Store the mask in the dictionary with the class name
            class_id = int(results[0].boxes.cls[i].cpu().numpy())
            class_name = model.names[class_id]

            # If the class already exists, append the mask; otherwise, create a new entry
            if class_name not in class_mask_dict:
                class_mask_dict[class_name] = [mask_np]
            else:
                class_mask_dict[class_name].append(mask_np)

    else:
        print("No masks found in the result.")

    return class_mask_dict


def plt_mask(dict: dict) -> None:
    """Plot the masks in the dictionary.

    Args:
        dict (dict): Dictionary containing class names as keys and masks as values.
    """
    for class_name, masks in dict.items():
        fig, axs = plt.subplots(1, len(masks), figsize=(5 * len(masks), 5))

        # Handle case where there is only one mask
        if len(masks) == 1:
            axs = [axs]

        print(f"Class: {class_name}, Number of masks: {len(masks)}")
        for j, mask in enumerate(masks):
            axs[j].imshow(mask, cmap='gray')
            axs[j].axis('off')
            axs[j].set_title(f'{class_name} - Mask {j+1}')

        plt.tight_layout()
        plt.show()
