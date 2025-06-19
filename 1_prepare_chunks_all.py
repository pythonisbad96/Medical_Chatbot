# 1.원천 데이터 폴더의 모든 json 파일을 불러오기
# 2.각 파일의 "content" 값을 '문장' 단위로 분리하고
# 3.3~5 문장씩 묶어 하나의 "chunk"로 만듬.
# 4.추후 '벡터 임베딩'에 활용하기 위해 all_chunks.json 으로 저장.

# 1_prepare_chunks_all.py 원천 데이터 청크 전처리

import os #폴더에서 파일 목록 가져오기
import json #.json파일 읽거나 저장
import re # 정규표현식 모듈로, 문장 "끝"을 기준으로 나누기 위해 사용.

raw_folder = r"D:\medical_knowledge_QA\Training\raw"
all_chunks = []

for filename in os.listdir(raw_folder): 
    if filename.endswith(".json"):
        file_path = os.path.join(raw_folder, filename)
        
        with open(file_path, "r", encoding = "utf-8-sig") as f:
            data = json.load(f)
            
        text = data.get("content", "") #JSON에서 "content"라는 key값을 꺼내고, "content"가 없으면 기본값 ""(빈 문자열)반환
        
        if not text.strip():
            continue
        
        sentences = re.split(r'(?<=[.?!])\s+', text)
        
        chunk_size = 4 #한 청크에 4문장을 넣음.
        
        for i in range(0, len(sentences), chunk_size):
            chunk_text = " ".join(sentences[i:i+chunk_size])
            
            chunk_info = {
                "source_file" : filename,
                "chunk_id" : f"{filename.replace('.json', '')}_{i// chunk_size + 1}",
                "text" : chunk_text
            }
            
            all_chunks.append(chunk_info)
            
with open("all_chunks.json", "w", encoding="utf-8-sig")as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
print(f"총 {len(all_chunks)}개의 청크가 저장되었습니다.")