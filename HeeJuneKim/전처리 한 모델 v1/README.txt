#전처리 안한 모델(LGAI-EXAONE/EXAONE-4.0-1.2B)

#학습 데이터 
- 1200_v1.csv 
- final_v7.csv

#실행 코드
- train.py (모델 학습)
- model_load.py (추론 테스트)

#모델 저장
- lora_exaone_output (학습 중간 결과, 체크포인트 저장)
- lora_exaone_adapter (최종 학습된 모델 배포용)

#백업
- model_load copy.py(추론 테스트 백업)

ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

#실행 순서
1. train.py
2. model_load.py

