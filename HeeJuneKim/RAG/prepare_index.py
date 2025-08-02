# ✅ 문서들을 SBERT로 임베딩하고 FAISS에 저장
from sentence_transformers import SentenceTransformer
import pandas as pd, faiss, pickle

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 여러 파일 병합
dfs = []
for file in ["final_v7.csv", "1200_v1.csv", "amc_cleaned.csv", "kdca_cleaned.csv", "snu_cleaned.csv"]:
    df = pd.read_csv(f"./symptom_disease_data/{file}")
    df = df.rename(columns={"text": "sentence"})  # 열 이름 통일
    dfs.append(df[["sentence"]])
symptom_df = pd.concat(dfs)

# 벡터화
vectors = model.encode(symptom_df["sentence"].tolist(), show_progress_bar=True)

# FAISS index 생성
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# 저장
faiss.write_index(index, "symptom_index.faiss")
with open("symptom_meta.pkl", "wb") as f:
    pickle.dump(symptom_df["sentence"].tolist(), f)
