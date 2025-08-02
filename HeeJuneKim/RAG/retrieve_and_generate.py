import faiss
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from rapidfuzz import fuzz  # ✅ 유사도 계산용

# SBERT 모델과 EXAONE text-generation 파이프라인 로딩
retriever = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
generator = pipeline(
    "text-generation",
    model="LGAI-EXAONE/EXAONE-4.0-1.2B",
    device_map="auto",
    trust_remote_code=True
)

# FAISS 인덱스 및 메타데이터 로딩
symptom_index = faiss.read_index("symptom_index.faiss")
with open("symptom_meta.pkl", "rb") as f:
    symptom_texts = pickle.load(f)

#-----------------------------------
# 헬퍼 함수들
#-----------------------------------

# 정확히 일치하는 증상 텍스트가 있으면 해당 병명을 반환, 없으면 "알 수 없음"
def get_related_disease(symptom_text):
    for file in ["final_v7.csv","1200_v1.csv","amc_cleaned.csv","kdca_cleaned.csv","snu_cleaned.csv"]:
        df = pd.read_csv(f"./symptom_disease_data/{file}")
        if symptom_text in df.get("text", []):
            return df.loc[df["text"] == symptom_text, "label"].values[0]
    return "알 수 없음"

# 유사도 기반으로 가장 높은 score를 가진 매핑 병명을 반환
def get_related_disease_fuzzy(user_symptom, threshold=80):
    best_score, best_label = 0, "알 수 없음"
    for file in ["final_v7.csv","1200_v1.csv","amc_cleaned.csv","kdca_cleaned.csv","snu_cleaned.csv"]:
        df = pd.read_csv(f"./symptom_disease_data/{file}")
        if "text" not in df.columns or "label" not in df.columns:
            continue
        for _, row in df.iterrows():
            score = fuzz.partial_ratio(user_symptom, row["text"])
            if score > best_score and score >= threshold:
                best_score, best_label = score, row["label"]
    return best_label

# 정확 일치가 없으면 fuzzy 매칭으로 fallback
def get_related_disease_with_fallback(symptom_text):
    label = get_related_disease(symptom_text)
    return label if label != "알 수 없음" else get_related_disease_fuzzy(symptom_text)

# 외부 문서에서 병명 관련 정보를 dict 형태로 반환, 없으면 빈 문자열
def get_disease_description(disease_name):
    for file in ["amc_rag.csv","kdca_rag.csv","snu_rag.csv"]:
        df = pd.read_csv(f"./disease_description_data/{file}")
        row = df[df["병명"].astype(str).str.contains(disease_name, na=False)]
        if not row.empty:
            return str(row.iloc[0].to_dict())
    return ""

#-----------------------------------
# RAG 기반 진단 파이프라인 함수
#-----------------------------------

# 사용자 입력을 벡터화하고 유사 증상 top-5를 찾아 컨텍스트 구성
def retrieve_context(user_input):
    vector = retriever.encode([user_input])
    D, I = symptom_index.search(vector, 15)
    candidates = [symptom_texts[i] for i in I[0]]
    seen, unique = set(), []
    for s in candidates:
        key = s.replace(" ", "").lower()[:30]
        if key not in seen:
            seen.add(key)
            unique.append(s)
        if len(unique) >= 5:
            break
    mapped = get_related_disease_with_fallback(user_input)
    doc = get_disease_description(mapped)
    return {
        "user_input": user_input,
        "symptoms": unique,
        "mapped": mapped,
        "doc_info": doc
    }

# few-shot 프롬프트 작성, EXAONE 호출 후 병명만 추출
def generate_response(context):
    examples = context["symptoms"]
    fewshot = "\n\n".join(
        f"증상: {s}\n병명: {get_related_disease_with_fallback(s)}"
        for s in examples
    )
    prompt = f"""
{fewshot}

당신은 전문 AI 의료 챗봇입니다.
사용자 증상: {context['user_input']}
사전 매핑된 병명: {context['mapped']}
외부 문서: {context['doc_info']}

병명만 출력하세요:
""".strip()
    output = generator(prompt, max_new_tokens=30)[0]["generated_text"]
    for line in reversed(output.split("\n")):
        clean = line.strip().replace("병명:", "").replace("입니다", "").replace(".", "")
        if 1 <= len(clean) <= 30:
            return clean
    return "알 수 없음"

# retrieve_context와 generate_response를 순차 실행하여 최종 진단 수행
def hybrid_diagnosis(user_input):
    ctx = retrieve_context(user_input)
    return generate_response(ctx)
