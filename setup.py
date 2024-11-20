from setuptools import setup, Extension

module = Extension(
    "fslic",  # Python 모듈 이름
    sources=["fslic.c"],  # 컴파일할 C 소스 파일
    include_dirs=[],  # 필요한 헤더 파일 디렉토리
)

setup(
    name="fslic",
    version="1.0",
    description="A Python wrapper for fslic C library",
    ext_modules=[module],
)