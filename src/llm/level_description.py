import pathlib
import google.generativeai as genai
from google.colab import files
import yaml

# API 키 설정
genai.configure(api_key='AIzaSyD1GX8f9XcpXQHQXEeC2b5MOM8t2N4qqxw')

# 모델 생성
model = genai.GenerativeModel('gemini-1.5-flash')

# 사용자로부터 이미지 파일 업로드 받기
uploaded = files.upload()

# 업로드한 파일의 경로 설정 (두 개의 이미지만 선택)
image_paths = list(uploaded.keys())[:2]

# 이미지 파일 로드 및 토큰 수 계산
images = []
total_image_token_count = 0
for image_path in image_paths:
    image_data = pathlib.Path(image_path).read_bytes()
    image_token_count = len(image_data) // 4  # 바이트를 4로 나눠 토큰 수 추정
    total_image_token_count += image_token_count
    images.append({
        'mime_type': 'image/png',  # 필요에 따라 이미지 타입을 변경 가능
        'data': image_data
    })

# 사용자로부터 단계 입력받기
level = input("원하는 설명 단계(1, 2, 3)를 입력하세요: ")

# YAML 형식의 가이드라인 생성
description_guide = {
    "Description_Prompt_Guide": {
        "Principles": {
            "description": "각 단계에 맞춰 이미지의 색상 및 분위기를 설명하는 가이드라인.",
            "guidelines": [
                "각 단계의 요구 사항을 충족하도록 설명의 구체성과 표현 스타일을 조절합니다.",
                "단계가 높아질수록 묘사에 감성적이거나 문학적인 표현을 더하여 색상과 분위기를 더 깊이 있게 전달합니다.",
                "각 단계는 색상, 명도, 채도, 위치 등 시각적인 요소를 점진적으로 풍부하게 묘사하도록 구성되어 있습니다."
            ]
        },
        "Description_Levels": [
            {
                "Level": "1단계: 간단한 이미지 설명",
                "Goal": "이미지에서 가장 눈에 띄는 주요 색상만 간단히 설명하여 기본적인 색상 정보를 제공합니다.",
                "Criteria": ["단순한 색상 언급", "명도, 채도 등 추가 설명 없이 주요 색상만 명시"],
                "Prompt": "이미지에서 눈에 띄는 주요 색상을 간단히 나열하여 설명해 주세요.",
                "Example_Output": "이미지 속 시바견들은 각각 흰색, 붉은색, 검은색, 그리고 흰색과 검은색이 섞인 삼색을 가진 모습입니다."
            },
            {
                "Level": "2단계: 약간의 비유가 추가된 설명",
                "Goal": "주요 색상의 명도와 채도, 그리고 각 색상이 위치한 곳을 구체적으로 설명하여 이미지를 더 명확히 전달합니다.",
                "Criteria": ["색상의 명도와 채도 언급", "이미지 내 특정 위치에 따른 색상 변화를 설명"],
                "Prompt": "이미지 속에서 색상의 명도, 채도, 그리고 위치를 포함하여 각 색상이 어떻게 분포되어 있는지 설명해 주세요.",
                "Example_Output": "왼쪽에 위치한 흰색 시바견은 부드럽고 밝은 톤을 가지고 있으며, 가운데 붉은색 시바견은 따뜻하고 활기찬 붉은빛을 띄고 있습니다. 세 번째에 있는 검은색 시바견은 깊고 어두운 톤을 가지고 있으며, 오른쪽 끝의 삼색 시바견은 검정, 흰색, 붉은색이 조화롭게 섞여 중간 명도를 형성하고 있습니다."
            },
            {
                "Level": "3단계: 감성적 또는 문학적 표현이 포함된 설명",
                "Goal": "색상 묘사에 감성적, 문학적 표현을 더하여 이미지에 대한 감정적 느낌과 분위기를 생동감 있게 전달합니다.",
                "Criteria": ["색상 묘사의 감성적, 문학적 표현 사용", "색상과 장면의 분위기 전달", "각 색상이 주는 인상이나 감정을 묘사"],
                "Prompt": "이미지 속 색상의 감성적 느낌과 분위기를 묘사하고, 각 색상이 주는 인상을 생생하게 표현해 주세요.",
                "Example_Output": """|
                    눈처럼 깨끗한 흰색 시바견은 순수하고 조용한 아름다움을 자랑하며,
                    붉은 저녁노을을 닮은 붉은색 시바견은 따스함과 활력을 보여줍니다.
                    신비로운 밤하늘처럼 어두운 빛을 띠는 검은색 시바견은 깊은 고요함과 카리스마를 느끼게 하며,
                    여러 색이 어우러진 삼색 시바견은 독특하면서도 조화로운 매력을 발산하며 시선을 사로잡습니다."""
            }
        ],
        "Output_Format": {
            "format": {
                "description": "1. input 이미지에 대한 전체적인 설명 2. object 마다 각각 색상 설명(색상을 기준으로 분류하지 않고 모든 object에 대해 설명), 1번 2번 각각 줄바꿈 구분하여 출력"
            },
            "example": [
                {
                    "level": "1단계: 간단한 이미지 설명",
                    "description": """|
                        이미지 속 시바견들은 크게 흰색, 붉은색, 검은색, 그리고 삼색으로 나눌 수 있습니다.
                        흰색 시바견은 털이 깨끗하고 밝습니다.
                        붉은색 시바견은 털빛이 짙고 선명합니다.
                        검은색 시바견은 털이 윤기 있고 촘촘합니다.
                        삼색 시바견은 흰색, 붉은색, 검은색 털이 조화롭게 섞여 있습니다."""
                },
                {
                    "level": "2단계: 약간의 비유가 추가된 설명",
                    "description": """|
                        각기 다른 매력을 가진 시바견들이 한자리에 모여 눈길을 사로잡습니다.
                        흰색 시바견은 순백의 눈송이를 닮았습니다.
                        붉은색 시바견은 붉은 저녁노을처럼 따뜻한 털빛을 가졌습니다.
                        검은색 시바견은 밤하늘의 별처럼 반짝입니다.
                        삼색 시바견은 그리고 봄날의 꽃밭처럼 화려합니다."""
                },
                {
                    "level": "3단계: 감성적 또는 문학적 표현이 포함된 설명",
                    "description": """|
                        마치 예술가의 팔레트 위에 놓인 물감처럼 다채로운 색깔을 자랑하는 시바견들의 모습은 보는 이들의 감탄을 자아냅니다.
                        순백의 털이 눈부시게 빛나는 흰색 시바견은 마치 겨울왕국의 여왕처럼 우아합니다.
                        붉은 불꽃을 연상케 하는 털빛을 가진 붉은색 시바견은 열정적인 모습을 보여줍니다.
                        신비로운 밤하늘을 연상시키는 검은색 시바견은 고혹적인 매력을 풍깁니다.
                        흰색, 붉은색, 검은색 털이 조화롭게 어우러진 삼색 시바견은 화려하고 독특한 아름다움을 자랑합니다."""
                }
            ]
        }
    }
}

# YAML 형식을 문자열로 변환
yaml_prompt = yaml.dump(description_guide)

# 선택한 단계에 맞는 프롬프트 추가
if level in ["1", "2", "3"]:
    selected_prompt = description_guide["Description_Prompt_Guide"]["Description_Levels"][int(level) - 1]["Prompt"]
    full_prompt = yaml_prompt + f"\n\n사용자 요청: {selected_prompt}"
else:
    print("잘못된 단계가 입력되었습니다. 1, 2, 3 중에서 선택해 주세요.")
    exit()

# 컨텐츠 생성 요청
response = model.generate_content([full_prompt] + images)

print("생성된 설명:")
print(response.text)
