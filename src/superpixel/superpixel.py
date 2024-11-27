import numpy as np
import fslic
from skimage import color


def superpixel(img_resized: np.ndarray, mask_dict: dict) -> np.ndarray:

    h, w, _ = img_resized.shape
    result_img = np.zeros_like(img_resized)

    for label, mask in mask_dict.items():

        masked_img = img_resized * mask[:, :, np.newaxis]

        img_lab = color.rgb2lab(masked_img)
        h, w, d = img_lab.shape
        img_flat = img_lab.reshape(-1).tolist()

        clusters = 100
        compactness = 100

        result = np.array(fslic.fslic(img_flat, w, h, d,
                          clusters, compactness, 10, 1, 1))
        result = result.reshape(h, w, d)

        result_rgb = color.lab2rgb(result)
        result_img += result_rgb * mask[:, :, np.newaxis]

    done_img = result_img.clip(0, 1)

    return done_img


def superpixel_no_mask(img_resized: np.ndarray) -> np.ndarray:

    img_lab = color.rgb2lab(img_resized)
    h, w, d = img_lab.shape
    img_flat = img_lab.reshape(-1).tolist()

    clusters = 100
    compactness = 100

    result = np.array(fslic.fslic(img_flat, w, h, d,
                      clusters, compactness, 10, 1, 1))
    result = result.reshape(h, w, d)
    result_rgb = color.lab2rgb(result)

    done_img = result_rgb.clip(0, 1)

    return done_img
