import numpy as np
import cv2
from skimage import measure

def remove_shadow(org_image: np.ndarray,
                  ab_threshold: int = 256,
                  lab_adjustment: bool = False,
                  region_adjustment_kernel_size: int = 10,
                  shadow_dilation_iteration: int = 10,
                  shadow_dilation_kernel_size: int = 5,
                  shadow_size_threshold: int = 2500,
                  verbose: bool = False) -> np.ndarray:
    

    def median_filter(img: np.ndarray, point: np.ndarray, filter_size: int) -> list:
        indices = [[x, y]
                   for x in range(point[1] - filter_size // 2, point[1] + filter_size // 2 + 1)
                   for y in range(point[0] - filter_size // 2, point[0] + filter_size // 2 + 1)]
        indices = list(filter(lambda x: not (x[0] < 0 or x[1] < 0 or
                                             x[0] >= img.shape[0] or
                                             x[1] >= img.shape[1]), indices))
        pixel_values = [0, 0, 0]
        for channel in range(3):
            pixel_values[channel] = list(img[index[0], index[1], channel] for index in indices)
        pixel_values = list(np.median(pixel_values, axis=1))
        return pixel_values

    def edge_median_filter(img: np.ndarray, contours_list: tuple, filter_size: int = 7) -> np.ndarray:
        temp_img = np.copy(img)
        for partition in contours_list:
            for point in partition:
                temp_img[point[0][1]][point[0][0]] = median_filter(img, point[0], filter_size)
        return cv2.cvtColor(temp_img, cv2.COLOR_HSV2BGR)

    def correct_region_lab(org_img: np.ndarray, shadow_clear_img: np.ndarray,
                           shadow_indices: np.ndarray, non_shadow_indices: np.ndarray) -> np.ndarray:
        shadow_average_lab = np.mean(org_img[shadow_indices[0], shadow_indices[1], :], axis=0)
        border_average_lab = np.mean(org_img[non_shadow_indices[0], non_shadow_indices[1], :], axis=0)
        lab_ratio = border_average_lab / shadow_average_lab
        shadow_clear_img = cv2.cvtColor(shadow_clear_img, cv2.COLOR_BGR2LAB)
        shadow_clear_img[shadow_indices[0], shadow_indices[1]] = np.uint8(
            shadow_clear_img[shadow_indices[0], shadow_indices[1]] * lab_ratio)
        shadow_clear_img = cv2.cvtColor(shadow_clear_img, cv2.COLOR_LAB2BGR)
        return shadow_clear_img

    def correct_region_bgr(org_img: np.ndarray, shadow_clear_img: np.ndarray,
                           shadow_indices: np.ndarray, non_shadow_indices: np.ndarray) -> np.ndarray:
        shadow_average_bgr = np.mean(org_img[shadow_indices[0], shadow_indices[1], :], axis=0)
        border_average_bgr = np.mean(org_img[non_shadow_indices[0], non_shadow_indices[1], :], axis=0)
        bgr_ratio = border_average_bgr / shadow_average_bgr
        shadow_clear_img[shadow_indices[0], shadow_indices[1]] = np.uint8(
            shadow_clear_img[shadow_indices[0], shadow_indices[1]] * bgr_ratio)
        return shadow_clear_img

    def process_regions(org_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        lab_img = cv2.cvtColor(org_image, cv2.COLOR_BGR2LAB)
        shadow_clear_img = np.copy(org_image)
        labels = measure.label(mask)
        non_shadow_kernel_size = (shadow_dilation_kernel_size, shadow_dilation_kernel_size)
        non_shadow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, non_shadow_kernel_size)
        CHANNEL_MAX = 255
        for label in np.unique(labels):
            if label == 0:
                continue
            temp_filter = np.zeros(mask.shape, dtype="uint8")
            temp_filter[labels == label] = CHANNEL_MAX
            if cv2.countNonZero(temp_filter) >= shadow_size_threshold:
                shadow_indices = np.where(temp_filter == CHANNEL_MAX)
                non_shadow_temp_filter = cv2.dilate(temp_filter, non_shadow_kernel,
                                                    iterations=shadow_dilation_iteration)
                non_shadow_temp_filter = cv2.bitwise_xor(non_shadow_temp_filter, temp_filter)
                non_shadow_indices = np.where(non_shadow_temp_filter == CHANNEL_MAX)
                contours, _ = cv2.findContours(temp_filter, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if lab_adjustment:
                    shadow_clear_img = correct_region_lab(lab_img, shadow_clear_img,
                                                          shadow_indices, non_shadow_indices)
                else:
                    shadow_clear_img = correct_region_bgr(org_image, shadow_clear_img,
                                                          shadow_indices, non_shadow_indices)
                shadow_clear_img = edge_median_filter(cv2.cvtColor(shadow_clear_img, cv2.COLOR_BGR2HSV),
                                                      contours)
        return shadow_clear_img

    def calculate_mask(org_image: np.ndarray) -> np.ndarray:
        lab_img = cv2.cvtColor(org_image, cv2.COLOR_BGR2LAB)
        l_range = (0, 100)
        ab_range = (-128, 127)
        lab_img = lab_img.astype('int16')
        lab_img[:, :, 0] = lab_img[:, :, 0] * l_range[1] / 255
        lab_img[:, :, 1] += ab_range[0]
        lab_img[:, :, 2] += ab_range[0]
        means = [np.mean(lab_img[:, :, i]) for i in range(3)]
        thresholds = [means[i] - (np.std(lab_img[:, :, i]) / 3) for i in range(3)]
        if sum(means[1:]) <= ab_threshold:
            mask = cv2.inRange(lab_img, (l_range[0], ab_range[0], ab_range[0]),
                                       (thresholds[0], ab_range[1], ab_range[1]))
        else:
            mask = cv2.inRange(lab_img, (l_range[0], ab_range[0], ab_range[0]),
                                       (thresholds[0], ab_range[1], thresholds[2]))
        kernel_size = (region_adjustment_kernel_size, region_adjustment_kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, mask)
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, mask)
        return mask

    # 그림자 마스크 계산
    mask = calculate_mask(org_image)
    # 그림자 영역 처리
    shadow_clear_img = process_regions(org_image, mask)
    # RGB 형식으로 반환
    return shadow_clear_img