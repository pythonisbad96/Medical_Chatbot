# -*- coding: utf-8 -*-
"""
medical_main.py (V4.5: 출력 간소화)
- 터미널에 출력되던 1차 검색 결과(유사도)와 LLM 컨텍스트 미리보기를 제거.
- 사용자에게는 AI의 질문과 최종 답변만 보이도록 수정.
"""

# ------------------------------------------------------------
# 0) 표준/서드파티 모듈 임포트
# ------------------------------------------------------------
import os, json, re
from typing import List, Tuple

# LangChain, OpenAI 등 라이브러리 임포트
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# .env 파일 로드 (선택)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# GPU 설정
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"


# ------------------------------------------------------------
# 1) 경로/DB 기본 설정
# ------------------------------------------------------------
JSON_FOLDER   = os.getenv("JSON_FOLDER", "./json_diseases_final_ver").strip()
_default_dbdir = f"vector_unified_{os.path.basename(JSON_FOLDER) or 'db'}"
DB_DIR         = os.getenv("DB_DIR", _default_dbdir).strip()

UNIFIED_DB_PATH  = f"{DB_DIR}/faiss_unified_disease_db"
os.makedirs(DB_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2) 실행 옵션 및 기준값(Threshold) 설정
# ------------------------------------------------------------
FORCE_REBUILD    = os.getenv("FORCE_REBUILD", "0") == "1"
K_DISEASE    = int(os.getenv("K_DISEASE", "10"))
MAX_DISEASES = int(os.getenv("MAX_DISEASES", "5"))
CTX_CHARS    = int(os.getenv("CTX_CHARS", "4000"))

LOW_CONF_THRESHOLD = 0.5
HIGH_CONF_THRESHOLD = 0.74
SCORE_DIFF_THRESHOLD = 0.03


# ------------------------------------------------------------
# 3) 임베딩 모델 및 A.X 클라이언트 준비
# ------------------------------------------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask").strip()
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
)

AX_BASE_URL = os.getenv("AX_BASE_URL", "https://guest-api.sktax.chat/v1").strip()
AX_API_KEY  = os.getenv("ADOTX_API_KEY", "").strip()
AX_MODEL    = os.getenv("AX_MODEL", "ax4").strip()
if not AX_API_KEY:
    raise RuntimeError("환경변수 ADOTX_API_KEY가 비어 있습니다.")
client = OpenAI(base_url=AX_BASE_URL, api_key=AX_API_KEY)


# ------------------------------------------------------------
# 4) LLM 호출 및 유틸리티 함수
# ------------------------------------------------------------
def chat_with_ax(messages, **gen_opts):
    completion = client.chat.completions.create(
        model=AX_MODEL,
        messages=messages,
        temperature=gen_opts.get("temperature", 0.3),
        max_tokens=gen_opts.get("max_tokens", 1024),
    )
    return completion.choices[0].message.content.strip()

def extract_any(section_val) -> str:
    if section_val is None: return ""
    if isinstance(section_val, dict):
        texts = []
        if 'supplement' in section_val:
            supp = section_val.get('supplement')
            if supp and isinstance(supp, list): texts.append("\n".join(str(x) for x in supp if x and str(x) != "None"))
        for k, v in section_val.items():
            if k == 'supplement': continue
            if isinstance(v, list): texts.append("\n".join(str(x) for x in v if x and str(x) != "None"))
            elif v and str(v) != "None": texts.append(str(v))
        return "\n".join(texts).strip()
    if isinstance(section_val, list): return "\n".join(str(x) for x in section_val if x and str(x) != "None").strip()
    return str(section_val).strip()

def get_disease_from_doc(doc):
    return getattr(doc, "metadata", {}).get("병명", "알 수 없는 질병")

def extract_numbered_block(answer: str) -> str:
    items = re.findall(r'(?ms)^\s*\d\.\s.*?(?=^\s*\d\.|\Z)', answer)
    items = [it.strip() for it in items if it.strip()]
    if not items: return answer.strip()
    if len(items) >= 6:   keep = items[:6]
    elif len(items) >= 5: keep = items[:5]
    elif len(items) >= 3: keep = items[:3]
    else:                 keep = items
    return "\n".join(keep).strip()


# ------------------------------------------------------------
# 5) FAISS 헬퍼 및 단일 인덱스 구축 함수
# ------------------------------------------------------------
def faiss_from_texts(texts, embedding_model, metadatas=None):
    try: return FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    except TypeError: return FAISS.from_texts(texts, embeddings=embedding_model, metadatas=metadatas)

def faiss_load_local(path, embedding_model):
    try: return FAISS.load_local(path, embedding=embedding_model, allow_dangerous_deserialization=True)
    except TypeError: return FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def _index_file(path: str) -> str: return os.path.join(path, "index.faiss")
def _latest_json_mtime(folder: str) -> float:
    times = []
    if not os.path.isdir(folder): return 0.0
    for f in os.listdir(folder):
        if f.lower().endswith(".json"): times.append(os.path.getmtime(os.path.join(folder, f)))
    return max(times) if times else 0.0
def _needs_rebuild(index_path: str, source_folder: str) -> bool:
    if FORCE_REBUILD: return True
    idx = _index_file(index_path)
    if not os.path.exists(idx): return True
    return os.path.getmtime(idx) < _latest_json_mtime(source_folder)

def build_or_load_unified_disease_db():
    if _needs_rebuild(UNIFIED_DB_PATH, JSON_FOLDER):
        print("[Rebuild] 통합 질병 인덱스 (검색용/LLM용 분리)를 새로 생성합니다.")
        texts_for_embedding, metas = [], []
        files = sorted([f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")])
        
        for filename in files:
            with open(os.path.join(JSON_FOLDER, filename), encoding="utf-8") as f: data = json.load(f)
            
            disease_name = (data.get("병명") or "").strip()
            if not disease_name: continue

            symptom_data = data.get("증상", {})
            symptom_text = extract_any(symptom_data)
            
            supplement_text = ""
            if isinstance(symptom_data, dict) and 'supplement' in symptom_data:
                supplement_text = extract_any(symptom_data.get('supplement'))

            weighted_symptom_part = (f"[증상] {symptom_text}\n" + f"[증상.supplement] {supplement_text}\n") * 3

            other_info_parts = [f"[병명] {disease_name}"]
            for key, value in data.items():
                if key not in ["병명", "증상"]:
                    content = extract_any(value)
                    if content: other_info_parts.append(f"[{key}] {content}")
            other_info_part = "\n".join(other_info_parts)
            
            weighted_document_text = (weighted_symptom_part + other_info_part).strip()
            texts_for_embedding.append(weighted_document_text)

            clean_symptom_part = f"[증상] {symptom_text}"
            clean_document_text = (clean_symptom_part + "\n" + other_info_part).strip()
            
            metas.append({"병명": disease_name, "파일": filename, "clean_text": clean_document_text})

        if not texts_for_embedding: raise RuntimeError("통합 인덱스를 만들 텍스트가 없습니다.")
        
        db = faiss_from_texts(texts_for_embedding, embedding_model, metadatas=metas)
        db.save_local(UNIFIED_DB_PATH); return db
    else:
        print("[Load] 기존 통합 질병 인덱스를 불러옵니다.")
        return faiss_load_local(UNIFIED_DB_PATH, embedding_model)

# ------------------------------------------------------------
# 6) 단일 검색 함수
# ------------------------------------------------------------
def search_unified_db_with_scores(db, user_query: str, k: int) -> List[Tuple[any, float]]:
    q_fmt = f"query: {user_query}"
    if not db: return []
    return db.similarity_search_with_score(q_fmt, k)


# ------------------------------------------------------------
# 7) 메인 루프
# ------------------------------------------------------------
SYSTEM_PROMPT = """
당신은 의료 상담 챗봇입니다.
사용자 질문이 건강/증상/의학 관련이면, 아래 [질병 정보]를 참고하여 '출력 형식'에 맞춰 답변하세요.
'상비약 추천'은 당신의 의료 지식을 바탕으로 답변해야 합니다.
불필요한 서론/결론 없이 '출력 형식'의 항목만 간결하게 답변하세요.

출력 형식:
1. 예상되는 병명 (2~3가지):
   - 첫 번째 병명은 **굵게** 표기하고 간단한 설명도 포함하세요.
2. 주요 원인:
3. 추천 진료과 (2~3과):
4. 예방 및 관리 방법:
5. 생활 시 주의사항:
6. 상비약 추천(실제 제품):
""".strip()

if __name__ == "__main__":
    disease_db = build_or_load_unified_disease_db()
    print("\n✅ 통합 인덱스 준비 완료")

    while True:
        user_input = input("\n🩺 증상을 입력하세요 (종료: exit): ").strip()
        if user_input.lower() in ["exit", "종료", "quit"]:
            print("챗봇을 종료합니다."); break
        
        docs_with_scores = search_unified_db_with_scores(disease_db, user_input, k=K_DISEASE)
        
        if not docs_with_scores:
            print("[Info] 관련 질병 정보를 찾을 수 없습니다. 일반적인 답변을 시도합니다.")
            general_messages = [{"role": "system", "content": "당신은 사용자에게 친절하게 답변하는 AI 어시스턴트입니다."}, {"role": "user", "content": user_input}]
            try:
                answer = chat_with_ax(general_messages)
                print(f"\n🧾 [A.X 4.0 답변]\n{answer}")
            except Exception as e: print(f"[오류] API 호출 실패: {e}")
            continue

        docs_with_sim_scores = [(doc, 1 / (1 + score)) for doc, score in docs_with_scores]
        unique_docs = docs_with_sim_scores
        
        # ★★★ 수정된 부분: 유사도 점수 출력을 주석 처리 ★★★
        # print("\n--- [1차 검색 결과 (Top 3)] ---")
        # for doc, score in unique_docs[:3]:
        #     print(f"  - 유사도: {score:.4f} / 질병명: {get_disease_from_doc(doc)}")
        # print("---------------------------------")
        
        top1_doc, top1_score = unique_docs[0]
        final_docs = []

        if top1_score < LOW_CONF_THRESHOLD:
            print(f"[판단] 비의료 질문 (유사도: {top1_score:.2f} < {LOW_CONF_THRESHOLD})")
            general_messages = [{"role": "system", "content": "당신은 사용자에게 친절하게 답변하는 AI 어시스턴트입니다."}, {"role": "user", "content": user_input}]
            try:
                answer = chat_with_ax(general_messages)
                print(f"\n🧾 [A.X 4.0 답변]\n{answer}")
            except Exception as e:
                print(f"[오류] API 호출 실패: {e}")
            continue

        is_confident = top1_score >= HIGH_CONF_THRESHOLD and \
                       (len(unique_docs) < 3 or (unique_docs[0][1] - unique_docs[2][1]) >= SCORE_DIFF_THRESHOLD)
        
        if is_confident:
            print(f"[판단] 확신도 높음 (유사도: {top1_score:.2f})")
            final_docs = [doc for doc, score in unique_docs[:MAX_DISEASES]]
        else:
            print(f"[판단] 확신도 낮음 (유사도: {top1_score:.2f}). 추가 증상을 요청합니다.")
            print(f"\n증상을 조금 더 구체적으로 알려주시겠어요? 추가적인 증상이 있다면 함께 입력해주세요.")
            user_answer = input("[추가 증상 입력]: ").strip()
            
            if user_answer:
                combined_input = f"{user_input}\n추가 정보: {user_answer}"
                print("\n[Info] 추가 정보를 바탕으로 다시 검색합니다...")
                final_search_res = search_unified_db_with_scores(disease_db, combined_input, k=K_DISEASE)
                final_docs = [doc for doc, score in final_search_res[:MAX_DISEASES]]
            else:
                print("[Info] 추가 입력이 없어 초기 검색 결과로 답변을 생성합니다.")
                final_docs = [doc for doc, score in unique_docs[:MAX_DISEASES]]

        if final_docs:
            final_context = "\n---\n".join([doc.metadata.get('clean_text', doc.page_content) for doc in final_docs])
            final_user_input = user_input
            if 'combined_input' in locals() and 'user_answer' in locals() and user_answer:
                final_user_input = combined_input

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"[질병 참고]\n{final_context[:CTX_CHARS]}"},
                {"role": "user", "content": final_user_input},
            ]
            
            # ★★★ 수정된 부분: 컨텍스트 미리보기 출력을 주석 처리 ★★★
            # print("\n🧩 [LLM 컨텍스트 미리보기 (클린 버전)]")
            # print(final_context[:CTX_CHARS])
            # print("-" * 60)
            
            try:
                answer = chat_with_ax(messages)
                print("\n🧾 [A.X 4.0 최종 답변]")
                print(extract_numbered_block(answer))
            except Exception as e:
                print(f"[오류] 최종 답변 생성 실패: {e}")