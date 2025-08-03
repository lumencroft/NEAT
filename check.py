# check_neat_config_path.py
import neat
import os

# 라이브러리가 설치된 디렉토리에서 기본 설정 파일의 경로를 찾습니다.
try:
    config_path = os.path.join(os.path.dirname(neat.__file__), 'DefaultGenome.config')
    print(f"기본 설정 파일의 위치: {config_path}")
    print("\n이 파일의 내용을 복사하여 당신의 설정 파일에 붙여넣으세요.")
except Exception as e:
    print(f"경로를 찾는 중 오류 발생: {e}")