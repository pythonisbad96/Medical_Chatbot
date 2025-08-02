폴더 구조

- disease_description_data -> 병명관련정보
- symptom_disease_data -> 질병 증상 맵핑 csv 

파일 설명

- prepare_index.py -> 질병, 증상 맵핑 데이터 벡터화 임베딩 & 저장

- retrieve_and_generate -> 외부 문서 참고 및 retrieve and 답변 생성

- streamlit_app.py -> UI 실행

- symptom_index.faiss -> 학습용 데이터 벡터 인덱스

- symptom_meta.pkl -> 벡터들이 어떤 문장에 대응하는지 매핑해주는 메타 데이터 파일






실행 순서

1. prepare_index.py 

2. retrieve_and_generate.py

3. streamlit_app.py 