import yaml
import numpy as np
import base64
from PIL import Image
import io
from openai import OpenAI

class description_with_gpt:
    def __init__(self, yaml_path):
        self.client = OpenAI(api_key='')  # API 키 입력
        self.description_guide = self.load_yaml(yaml_path)
    
    def load_yaml(self, yaml_path):
        """Load YAML configuration file."""
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_description_guide(self, level):
        """Retrieve the description guide for a specific level."""
        return self.description_guide.get(f"level_{level}")
    
    def processing_image_with_level(self, img_arrays, level):
        if isinstance(img_arrays, np.ndarray):
            img_arrays = [img_arrays[i] for i in range(img_arrays.shape[0])]
        
        # Prepare base64 encoded images
        base64_images = []
        for img in img_arrays:
            img_pil = Image.fromarray(img)
            buffered = io.BytesIO()
            img_pil.save(buffered, format="JPEG")
            base64_images.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

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
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}} 
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

        print("\nGenerated Description:")
        print(response.choices[0].message.content)

        return response.choices[0].message.content


if __name__ == "__main__":
    # Load images and convert to numpy arrays
    img1 = Image.open('1.jpg')
    img2 = Image.open('2.jpg')
    img_arrays = np.array([np.array(img1), np.array(img2)])
    
    level = 3
    yaml_path = 'description_guide.yaml'

    pc1 = description_with_gpt(yaml_path)
    result = pc1.processing_image_with_level(img_arrays, level)
