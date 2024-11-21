import numpy as np
import base64
from PIL import Image
import io
from openai import OpenAI

class description_with_gpt:
  def __init__(self):
    self.client = OpenAI(api_key='') # API키 입력
    self.description_guide = {
      "level_1": {
          "Description_Prompt_Guide": {
              "instruction": {
                  "description": "Guidelines for describing the image's colors and atmosphere at each stage.",
                  "Level": "Level 1: Simple Image Description",
                  "Goal": "Focus on describing the most prominent objects in the image with color expressions.",
                  "Criteria": ["Mention simple colors", "Additional explanation of brightness, saturation, etc."],
                  "Prompt": "Please briefly list the prominent colors in the image.",
                  "Example_Output": """The Shiba Inu dogs can be largely divided into white, red, black, and tricolor.
  The white Shiba Inu has clean and bright fur.
  The red Shiba Inu has deep and vivid fur color.
  The black Shiba Inu has glossy and dense fur.
  The tricolor Shiba Inu has white, red, and black fur harmoniously mixed."""
              },
              "input_format": ["Original image in numpy format"],
              "Output_Format": ["1. Overall description of the input image 2. Color description for each object (describe all objects without classifying by color), output 1 and 2 separated by line break"]
          }
      },
      "level_2": {
          "Description_Prompt_Guide": {
              "instruction": {
                  "description": "Guidelines for describing the image's colors and atmosphere at each stage.",
                  "Level": "Level 2: Description with Emotional or Literary Expression",
                  "Goal": "Add emotional and literary expressions to color descriptions to vividly convey the image's emotional feeling and atmosphere.",
                  "Criteria": ["Use emotional and literary expressions in color description", "Convey color and scene atmosphere", "Describe the impression or emotion each color gives"],
                  "Prompt": "Describe the emotional feeling and atmosphere of colors in the image, and express the impression of each color vividly.",
                  "Example_Output": """The Shiba Inu dogs, boasting colors like paints on an artist's palette, evoke admiration from viewers.
  The white Shiba Inu with fur shining brilliantly is as elegant as the queen of the Winter Kingdom.
  The red Shiba Inu with fur color reminiscent of a flame of passion shows a passionate appearance.
  The black Shiba Inu, reminiscent of a mysterious night sky, exudes a bewitching charm.
  The tricolor Shiba Inu with white, red, and black fur harmoniously blended boasts a brilliant and unique beauty."""
              },
              "input_format": ["Original image in numpy format and Superpixel-processed image"],
              "Output_Format": ["1. Overall description of the input image 2. Color description for each object (describe all objects without classifying by color), output 1 and 2 separated by line break"]
          }
      },
      "level_3": {
          "Description_Prompt_Guide": {
              "instruction": {
                  "description": "Guidelines for describing the image's colors and atmosphere at each stage.",
                  "Level": "Level 3: In-depth Literary Description",
                  "Goal": "Deeply and richly interpret the colors and atmosphere, revealing the image's essence through emotional and literary expressions.",
                  "Criteria": ["Very delicate and deep literary expression", "Convey subtle nuances of color and emotional depth"],
                  "Prompt": "Interpret the image's colors and atmosphere with the deepest and richest literary expression.",
                  "Example_Output": """The colors of the Shiba Inus are like a living canvas inspired by a poet's palette.
  The pure white fur captures a clean and sacred moment like the first snowlight of a pure winter morning.
  The passionate red fur erupts with the energy of a flame of passion, like the sunset of a hot summer day.
  The mysterious black fur holds a deep and eternal mystery, like the quietude of a deep night, a veil enveloping an unknown world.
  The harmonious blend of three colors reflects the principles of the universe, performing a symphonic harmony created by different colors."""
              },
              "input_format": ["Original image in numpy format and complex image analysis results"],
              "Output_Format": ["1. In-depth interpretation of the input image 2. Literary and emotional depth description of colors for each object, output 1 and 2 separated by line break"]
          }
      },
      "level_4": {
          "Description_Prompt_Guide": {
              "instruction": {
                  "description": "Guidelines for describing the image's colors and atmosphere at each stage.",
                  "Level": "Level 2: Description with Emotional or Literary Expression",
                  "Goal": "Add emotional and literary expressions to color descriptions to vividly convey the image's emotional feeling and atmosphere.",
                  "Criteria": ["Use emotional and literary expressions in color description", "Convey color and scene atmosphere", "Describe the impression or emotion each color gives"],
                  "Prompt": "Describe the emotional feeling and atmosphere of colors in the image, and express the impression of each color vividly.",
                  "Example_Output": """Like colors on an artist's palette, the Shiba Inus evoke admiration from viewers.
  The white Shiba Inu with fur shining brilliantly is as elegant as the queen of the Winter Kingdom.
  The red Shiba Inu with fur color reminiscent of a flame of passion shows a passionate appearance.
  The black Shiba Inu, reminiscent of a mysterious night sky, exudes a bewitching charm.
  The tricolor Shiba Inu with white, red, and black fur harmoniously blended boasts a brilliant and unique beauty."""
              },
              "input_format": ["Original image in numpy format and Superpixel-processed image"],
              "Output_Format": ["1. Overall description of the input image 2. Color description for each object (describe all objects without classifying by color), output 1 and 2 separated by line break"]
          }
      }
  }
    
  def get_description_guide(self, level):
      return self.description_guide.get(f"level_{level}")
    
  def processing_image_with_level(self, img_arrays, level):
      # Ensure img_arrays is a list or 2D numpy array of image arrays
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

      # Prepare prompt
      prompt_instruction = guide['Description_Prompt_Guide']['instruction']['Prompt']
      example_output = guide['Description_Prompt_Guide']['instruction']['Example_Output']

      # Construct full prompt
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
          max_tokens=1000  # Increased to accommodate multiple images
      )

      # Print result
      print("\nGenerated Description:")
      print(response.choices[0].message.content)

      return response.choices[0].message.content



########### For testing
if __name__ == "__main__":
    # Load images and convert to numpy arrays
    img1 = Image.open('1.jpg')
    img2 = Image.open('2.jpg')
    img_arrays = np.array([np.array(img1), np.array(img2)])
    
    level = 3

    pc1 = description_with_gpt()
    result = pc1.processing_image_with_level(img_arrays, level)