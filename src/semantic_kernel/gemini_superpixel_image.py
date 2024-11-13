# 2. 원본 이미지 + superpixel 처리 된 이미지를 넣는 경우
import pathlib
import google.generativeai as genai
from google.colab import files

# API 키 설정
genai.configure(api_key='') #API key notion(Simple memos/Gemini API test)에 올려둠

# Gemini Pro Vision 모델 생성
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

# 사용자로부터 프롬프트 입력받기
prompt = input("프롬프트를 입력하세요: ")

# 프롬프트 토큰 수 계산 (문자 수로 추정)
prompt_token_count = len(prompt)

# 토큰 수 출력
print(f"프롬프트 토큰 수: {prompt_token_count} 토큰")
print(f"이미지 토큰 수 (총합): {total_image_token_count} 토큰")

# 컨텐츠 생성 요청
response = model.generate_content([prompt] + images)

# 결과 출력
print(response.text)
