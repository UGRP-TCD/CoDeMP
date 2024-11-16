import base64
import yaml
from openai import OpenAI
from google.colab import files

# OpenAI API 키 설정
client = OpenAI(
    api_key=''
)

# 이미지 업로드
uploaded = files.upload()

# 업로드된 파일 경로 설정 (최대 2개 이미지)
image_paths = list(uploaded.keys())[:2]

# 이미지 파일 로드
images = []
for image_path in image_paths:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        images.append(base64_image)

# 설명 레벨 입력 받기
level = input("Enter the desired description level (1, 2, 3): ")

# YAML 형식의 설명 가이드
description_guide = """
Description_Prompt_Guide:
  Principles:
    - Adjust the specificity and style of the description to meet the requirements of each level.
    - For higher levels, add emotional or literary expressions to convey the colors and mood more deeply.
    - Each level is structured to progressively enrich the visual description of elements such as color, brightness, saturation, and location.
    - Focus solely on the content of the provided images. Do not include unrelated or fictional elements.
    - The prompt below is simply an example format for the output. Therefore, refer to the example prompt format and output an appropriate response to the input image.
  Description_Levels:
    - Level: "Level 1: Simple Color Description"
      Goal: "Briefly describe the main colors that stand out in the image for each object."
      Criteria: 
        - "Mention primary colors only"
        - "Separate descriptions for each object with line breaks"
      Example Prompt: |
        Provide a brief description of each object's main colors in the image, separated by line breaks:
        - Dog 1: Describe the main color of the first dog.
        - Dog 2: Describe the main color of the second dog.
        - Dog 3: Describe the main color of the third dog.
        - Dog 4: Describe the main color of the fourth dog.
    - Level: "Level 2: Descriptive with Analogies"
      Goal: "Describe the brightness, saturation, and position of colors using analogies."
      Criteria:
        - "Mention brightness and saturation"
        - "Use simple analogies and separate descriptions for each object"
      Example Prompt: |
        Describe each dog's colors using brightness and analogies, separating descriptions by line breaks:
        - Dog 1: Describe the first dog's color with some analogies about brightness and contrast.
        - Dog 2: Describe the second dog's color with details about how it blends or contrasts with others.
        - Dog 3: Describe the third dog's color with analogies that enhance its characteristics.
        - Dog 4: Describe the fourth dog's color using analogies to create a vivid scene.
    - Level: "Level 3: Emotional and Literary Description"
      Goal: "Add emotional and literary expressions to convey mood and impression."
      Criteria:
        - "Use emotional and literary expressions for colors"
        - "Convey the mood and impression of each object's colors"
      Example Prompt: |
        Provide a vivid, emotional description of each dog's colors and mood, separating descriptions by line breaks:
        - Dog 1: Describe the first dog's colors with a literary touch, adding emotional context to its appearance.
        - Dog 2: Describe the second dog's colors to evoke a specific feeling or character.
        - Dog 3: Describe the third dog's colors to bring out its distinct personality.
        - Dog 4: Describe the fourth dog's colors with a focus on creating a mood or atmosphere.
"""

# YAML 문자열을 파이썬 딕셔너리로 변환
guide = yaml.safe_load(description_guide)

# 선택된 레벨에 대한 프롬프트 가져오기
selected_level = guide['Description_Prompt_Guide']['Description_Levels'][int(level) - 1]
prompt_instruction = selected_level['Example Prompt']

# 실제 이미지에 대한 설명 요청 프롬프트 구성
full_prompt = f"""
Analyze the uploaded image carefully. 
Based on the description level {level}, provide a detailed color description:

{prompt_instruction}

Important Guidelines:
- Observe the actual colors in the image
- Provide specific descriptions for object
- Follow the selected description level precisely
"""

# 메시지 구성
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": full_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{images[0]}"}},
        ],
    }
]

# GPT-4 Vision API 호출
response = client.chat.completions.create(
    model="gpt-4o",  # Vision 모델 사용
    messages=messages,
    max_tokens=500  # 충분한 토큰 할당
)

# 결과 출력
print("\nGenerated Description:")
print(response.choices[0].message.content)