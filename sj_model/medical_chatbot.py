# medical_rag_pipeline.py

import os
import pandas as pd
import glob
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
import re
from IPython.display import display, Markdown

# 1. 데이터 폴더/경로 지정
data = "./dataset"

# 2. 데이터 불러오기
df_1200 = pd.read_csv(os.path.join(data, "1200_v1.csv"))
df_amc = pd.read_csv(os.path.join(data, "amc.csv"))
df_daily = pd.read_csv(os.path.join(data, "daily_dataset.csv"))
df_final = pd.read_csv(os.path.join(data, "final_v7.csv"))
df_kdca = pd.read_csv(os.path.join(data, "kdca.csv"))
df_snu = pd.read_csv(os.path.join(data, "snu.csv"))

# 3. 임베딩 모델 준비 (최초 1회)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")


# 4. chunking 함수 정의
def chunk_text(texts, chunk_size=300):
    chunks = []
    for text in texts:
        if pd.isnull(text):
            continue
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
    return chunks


# 5. 각 파일별 chunk + 벡터DB 저장 (최초 1회, 그 후엔 필요X)
def prepare_faiss():
    # 벡터DB 폴더 자동 생성
    os.makedirs("vector_db", exist_ok=True)

    # 1200_v1
    texts_1200 = df_1200["label"].astype(str) + "\n" + df_1200["text"].astype(str)
    chunks_1200 = chunk_text(texts_1200.tolist(), chunk_size=300)
    db_1200 = FAISS.from_texts(chunks_1200, embedding=embedding_model)
    db_1200.save_local("vector_db/faiss_db_1200_v1")

    # amc
    amc_texts = (
        df_amc["병명"].astype(str)
        + "\n"
        + df_amc["정의"].astype(str)
        + "\n"
        + df_amc["원인"].astype(str)
        + "\n"
        + df_amc["증상"].astype(str)
        + "\n"
        + df_amc["진단"].astype(str)
        + "\n"
        + df_amc["치료"].astype(str)
    )
    chunks_amc = chunk_text(amc_texts.tolist(), chunk_size=400)
    db_amc = FAISS.from_texts(chunks_amc, embedding=embedding_model)
    db_amc.save_local("vector_db/faiss_db_amc")

    # daily
    daily_texts = df_daily["증상"].astype(str) + "\n" + df_daily["일상말"].astype(str)
    chunks_daily = chunk_text(daily_texts.tolist(), chunk_size=250)
    db_daily = FAISS.from_texts(chunks_daily, embedding=embedding_model)
    db_daily.save_local("vector_db/faiss_db_daily_dataset")

    # final_v7
    final_texts = df_final["label"].astype(str) + "\n" + df_final["text"].astype(str)
    chunks_final = chunk_text(final_texts.tolist(), chunk_size=300)
    db_final = FAISS.from_texts(chunks_final, embedding=embedding_model)
    db_final.save_local("vector_db/faiss_db_final_v7")

    # kdca
    kdca_texts = (
        df_kdca["병명"].astype(str)
        + "\n"
        + df_kdca["정의"].astype(str)
        + "\n"
        + df_kdca["원인"].astype(str)
        + "\n"
        + df_kdca["증상"].astype(str)
        + "\n"
        + df_kdca["진단"].astype(str)
        + "\n"
        + df_kdca["치료"].astype(str)
    )
    chunks_kdca = chunk_text(kdca_texts.tolist(), chunk_size=400)
    db_kdca = FAISS.from_texts(chunks_kdca, embedding=embedding_model)
    db_kdca.save_local("vector_db/faiss_db_kdca")

    # snu
    snu_texts = (
        df_snu["병명"].astype(str)
        + "\n"
        + df_snu["정의"].astype(str)
        + "\n"
        + df_snu["원인"].astype(str)
        + "\n"
        + df_snu["증상"].astype(str)
        + "\n"
        + df_snu["진단/검사"].astype(str)
        + "\n"
        + df_snu["치료"].astype(str)
    )
    chunks_snu = chunk_text(snu_texts.tolist(), chunk_size=400)
    db_snu = FAISS.from_texts(chunks_snu, embedding=embedding_model)
    db_snu.save_local("vector_db/faiss_db_snu")


# # 최초 1회만 실행
prepare_faiss()

# 6. 벡터DB 로드 (매번)
db_paths = [
    "vector_db/faiss_db_1200_v1",
    "vector_db/faiss_db_amc",
    "vector_db/faiss_db_daily_dataset",
    "vector_db/faiss_db_final_v7",
    "vector_db/faiss_db_kdca",
    "vector_db/faiss_db_snu",
]
db_list = [
    FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    for db_path in db_paths
]

# 7~11번 반복부 전체를 while True로 감싸기
while True:
    user_question = input("질문을 입력하세요(종료: exit): ")
    if user_question.lower() in ["exit", "quit", "종료"]:
        print("프로그램을 종료합니다.")
        break

    # 8. 검색+context 생성
    top_k = 1
    retrieved_chunks = []
    for db in db_list:
        docs = db.similarity_search(user_question, k=top_k)
        retrieved_chunks.extend([doc.page_content for doc in docs])
    retrieved_context = "\n---\n".join(retrieved_chunks)[:600]

    # 9. 프롬프트 생성
    prompt = f"""
아래 환자 증상에 대해 반드시 1~5번 형식으로 딱 한 번만 출력하세요.  
1,5번 항목은 '-입니다.', '-합니다.'와 같은 존댓말 종결어미로 통일하며, 설명을 반복하거나 불필요한 줄바꿈 없이 자연스럽게 작성해주세요.  
특히 1번 항목에서 언급된 첫 번째 병명은 일반인도 이해할 수 있도록 간단하게 정의해 주세요.

1. 예상되는 병명(2~3가지) :  
2. 병명 정의 :  
3. 추천 진료과(2-3과 추천) : 
4. 예방 및 관리 방법(2줄 이내) :  
5. 기타 환자에게 필요한 정보(생활/주의사항/추가정보 포함) :  

[증상 정보]  
{retrieved_context}
"""

    # 10. SKT A.X-4.0 API로 답변 생성
    client = OpenAI(
        base_url="https://guest-api.sktax.chat/v1",
        api_key="sktax-XyeKFrq67ZjS4EpsDlrHHXV8it",
    )

    response = client.chat.completions.create(
        model="ax4",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip()

    # 11. 번호 답변만 추출 (1~5)
    matches = re.findall(
        r"1\..*?\n2\..*?\n3\..*?\n4\..*?\n5\..*(?=\n|$)", answer, flags=re.DOTALL
    )
    answer_only = matches[-1].strip() if matches else answer

    print("\n[AX-4.0 답변]\n" + answer_only)
