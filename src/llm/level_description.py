import yaml
import numpy as np
import base64
from PIL import Image
import io
from openai import OpenAI


class DescriptionWithGPT:
    def __init__(self, yaml_path, orig_image, sup_image=None):
        self.client = OpenAI(api_key='api_key')  # API 키 입력
        self.description_guide = self.load_yaml(yaml_path)
        self._orig_image = orig_image  # 원본 이미지
        self._sup_image = sup_image  # 보조 이미지

    def load_yaml(self, yaml_path):
        """Load YAML configuration file."""
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def get_description_guide(self, level):
        """Retrieve the description guide for a specific level."""
        return self.description_guide.get(f"level_{level}")

    def processing_image_with_level(self, level, name, iter_num):
        # Determine images to process based on level
        if level in [1, 3]:
            img_arrays = [self._orig_image]
        elif level in [2, 4]:
            if self._sup_image is None:
                raise ValueError(
                    "Supplementary image is required for levels 2 and 4.")
            img_arrays = [self._orig_image, self._sup_image]
        else:
            raise ValueError(f"Invalid level: {level}")

        # Prepare base64 encoded images
        base64_images = []
        for img in img_arrays:
            img_pil = Image.fromarray(img)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            base64_images.append(base64.b64encode(
                buffered.getvalue()).decode('utf-8'))

        # Get guide for the selected level
        guide = self.get_description_guide(level)
        if not guide:
            raise ValueError(f"No description guide found for level {level}")

        prompt_instruction = guide['Description_Prompt_Guide']['instruction']['Prompt']
        example_output = guide['Description_Prompt_Guide']['instruction']['Example_Output']

        full_prompt = f"""
        Analyze the uploaded images carefully. 
        You will receive multiple images to describe.
        Based on the description level {level}, provide a detailed color description:

        {prompt_instruction}

        Example Output:
        {example_output}

        Important Guidelines:
        - Observe the actual colors in the images
        - Provide specific descriptions for each object
        - Follow the selected description level precisely
        """

        # Compose messages with multiple images
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt}
                ] + [
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"}}
                    for base64_img in base64_images
                ],
            }
        ]

        # Call GPT-4 Vision API
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000
        )
        
        save_sup_image = self._sup_image
        save_sup_image[:, :, [0, 2]] = save_sup_image[:, :, [2, 0]]
        save_sup_image = Image.fromarray(save_sup_image)
        save_sup_image.save(f'C:/UGRP/CoDeMP-simple/results/{name}.jpg', 'JPEG')

        # Save the generated description to a .txt file
        #output_filename = f"C:/UGRP/CoDeMP-simple/results/{name[-16:-1]}_{level}_{iter_num}.txt"
        output_filename = f"C:/UGRP/CoDeMP-simple/results/{name}_{level}_{iter_num}.txt"
        with open(output_filename, "w", encoding="utf-8") as file:
            file.write(response.choices[0].message.content)

        print(f"/nGenerated description saved to {output_filename}")


# if __name__ == "__main__":
#     # Load images as numpy arrays
#     orig_image = np.array(Image.open('1.jpg'))
#     sup_image = np.array(Image.open('2.jpg'))  # Optional

#     level = 2  # Example level
#     yaml_path = 'description_guide.yaml'

#     pc1 = DescriptionWithGPT(yaml_path, orig_image, sup_image)
#     result = pc1.processing_image_with_level(level)
