from dataset import load_images_from_folder, load_single_image, normalize_image
from llm.level_description import DescriptionWithGPT
from program import get_yolo_superpixel


def process_single_image(iter_num: int, file_path: str) -> None:
    image = load_single_image(file_path)

    norm_image = normalize_image(image)

    processed_image = get_yolo_superpixel(norm_image, mask_true=1)

    prompt = "llm/description_guide.yaml"
    prompt_object = DescriptionWithGPT(prompt, norm_image, processed_image)

    name = file_path.replace("/dataset/", "").replace(".jpg", "")
    for level in range(1, 5):
        for num in range(1, iter_num + 1):
            prompt_object.processing_image_with_level(level, name, num)


def process_multi_images(iter_num: int, folder_path: str) -> None:
    images = load_images_from_folder(folder_path)
    norm_images = []
    for image in images:
        norm_images.append(normalize_image(image))

    processed_images = []
    for norm_image in norm_images:
        processed_images.append(get_yolo_superpixel(norm_image, mask_true=1))

    prompt = "llm/description_guide.yaml"

    for i in range(1, 101):
        prompt_object = DescriptionWithGPT(
            prompt, norm_images[i - 1], processed_images[i - 1])
        for level in range(1, 5):
            for num in range(1, iter_num + 1):
                prompt_object.processing_image_with_level(level, i, num)


def main():
    # 필요한 거 주석 해제해서 사용할 것
    process_single_image(iter_num=20, file_path="/dataset/.jpg")
    # process_multi_images(iter_num=20, folder_path="/dataset")


if __name__ == '__main__':
    main()
