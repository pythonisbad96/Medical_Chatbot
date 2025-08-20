# -*- coding: utf-8 -*-
"""
medical_main.py (V4.6: FriendliAI(EXAONE 32B)로 LLM 교체 + 출력 간소화 유지)
- 터미널에 출력되던 1차 검색 결과(유사도)와 LLM 컨텍스트 미리보기를 제거.
- 사용자에게는 AI의 질문과 최종 답변만 보이도록 유지.
- LLM 클라이언트: FriendliAI Serverless (OpenAI 호환) 사용.
"""

# ------------------------------------------------------------
# 0) 표준/서드파티 모듈 임포트
# ------------------------------------------------------------
import os, json, re                          # os: 경로/환경변수, json: 파일 읽기, re: 정규표현식
from typing import List, Tuple               # 타입 힌트용

# LangChain, OpenAI 등 라이브러리 임포트
from langchain_huggingface import HuggingFaceEmbeddings  # 허깅페이스 임베딩 모듈(신규 패키지)
from langchain_community.vectorstores import FAISS       # FAISS 벡터DB
from openai import OpenAI                                # OpenAI 호환 클라이언트(Friendli가 이 인터페이스 지원)

# .env 파일 로드 (선택)
try:
    from dotenv import load_dotenv
    load_dotenv()                         # .env를 읽어 환경변수로 등록(없어도 통과)
except Exception:
    pass

# GPU 설정: torch가 있으면 CUDA 사용, 없으면 CPU 사용
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

UNIFIED_DB_PATH  = f"{DB_DIR}/faiss_unified_disease_db"   # 통합 인덱스 저장 경로
os.makedirs(DB_DIR, exist_ok=True)                        # 폴더가 없으면 생성


# ------------------------------------------------------------
# 2) 실행 옵션 및 기준값(Threshold) 설정
# ------------------------------------------------------------
FORCE_REBUILD    = os.getenv("FORCE_REBUILD", "0") == "1"   # 1이면 항상 인덱스 재생성
K_DISEASE    = int(os.getenv("K_DISEASE", "10"))            # 검색 상위 k개 문서
MAX_DISEASES = int(os.getenv("MAX_DISEASES", "5"))          # LLM에 투입할 최대 문서 수
CTX_CHARS    = int(os.getenv("CTX_CHARS", "4000"))          # LLM에 전달할 컨텍스트 길이 제한(문자수)

LOW_CONF_THRESHOLD = 0.5     # 비의료/일반 응답 라우팅 기준
HIGH_CONF_THRESHOLD = 0.74   # 확신도 높음 기준
SCORE_DIFF_THRESHOLD = 0.03  # 상위 결과 간 점수 차이 기준


# ------------------------------------------------------------
# 3) 임베딩 모델 준비
# ------------------------------------------------------------
# 주의: E5 계열 모델을 쓸 경우 쿼리에 "query: " 접두사가 유리합니다(아래 검색 함수에서 적용).
#       현재 기본값은 roberta 멀티태스크지만, 필요 시 EMBED_MODEL_NAME만 바꾸면 됩니다.
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask").strip()
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,                           # 임베딩 모델 이름
    model_kwargs={"device": DEVICE},                       # GPU/CPU 설정
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},  # 정규화 + 배치 크기
)


# ------------------------------------------------------------
# 4) FriendliAI LLM 클라이언트 준비 (OpenAI 호환)
# ------------------------------------------------------------
# Friendli Serverless OpenAI-호환 엔드포인트 & 모델 식별자
FRIENDLI_BASE_URL = "https://api.friendli.ai/serverless/v1"
FRIENDLI_MODEL = os.getenv("FRIENDLI_MODEL", "LGAI-EXAONE/EXAONE-4.0.1-32B").strip()

# 토큰은 환경변수로 주입(.env 또는 OS 환경변수)
FRIENDLI_TOKEN = os.getenv("FRIENDLI_TOKEN")
if not FRIENDLI_TOKEN:
    # 실행 전 .env에 FRIENDLI_TOKEN=xxxx 추가 또는 OS 환경변수 등록 필요
    raise RuntimeError("환경변수 FRIENDLI_TOKEN 이(가) 비었습니다. .env 또는 OS 환경변수에 설정하세요.")

# OpenAI 호환 클라이언트 객체(이름만 OpenAI, 실제로 Friendli 서버와 통신)
llm_client = OpenAI(
    api_key=FRIENDLI_TOKEN,       # Friendli 발급 토큰
    base_url=FRIENDLI_BASE_URL,   # Friendli OpenAI 호환 서버 URL
)

def chat_with_friendli(messages, **gen_opts) -> str:
    """
    FriendliAI Chat Completions 호출 (OpenAI 호환)
    - messages: [{"role": "...", "content": "..."}]
    - gen_opts: temperature, max_tokens 등
    """
    completion = llm_client.chat.completions.create(
        model=FRIENDLI_MODEL,
        messages=messages,
        temperature=gen_opts.get("temperature", 0.3),
        max_tokens=gen_opts.get("max_tokens", 1024),
    )
    return completion.choices[0].message.content.strip()


# ------------------------------------------------------------
# 5) LLM 보조 유틸리티 함수들
# ------------------------------------------------------------
def extract_any(section_val) -> str:
    """
    JSON의 각 필드(문자열/리스트/딕셔너리)를 일관되게 문자열로 풀어내는 함수.
    - 증상.supplement가 있으면 우선 포함.
    - "None" 같은 쓰레기 문자열은 제거.
    """
    if section_val is None:
        return ""
    if isinstance(section_val, dict):
        texts = []
        # supplement가 있으면 먼저 펼침
        if 'supplement' in section_val:
            supp = section_val.get('supplement')
            if supp and isinstance(supp, list):
                texts.append("\n".join(str(x) for x in supp if x and str(x) != "None"))
        # 나머지 키/값도 펼침
        for k, v in section_val.items():
            if k == 'supplement':
                continue
            if isinstance(v, list):
                texts.append("\n".join(str(x) for x in v if x and str(x) != "None"))
            elif v and str(v) != "None":
                texts.append(str(v))
        return "\n".join(texts).strip()
    if isinstance(section_val, list):
        return "\n".join(str(x) for x in section_val if x and str(x) != "None").strip()
    return str(section_val).strip()

def get_disease_from_doc(doc):
    """ FAISS 검색 결과 문서에서 병명 메타데이터를 안전히 꺼내기 """
    return getattr(doc, "metadata", {}).get("병명", "알 수 없는 질병")

def extract_numbered_block(answer: str) -> str:
    """
    LLM의 전체 답변 중 '1. ... 2. ...' 형태로 시작하는 번호 목록만 깔끔히 추출.
    항목이 너무 많으면 3~6개로 제한하여 간결화.
    """
    items = re.findall(r'(?ms)^\s*\d\.\s.*?(?=^\s*\d\.|\Z)', answer)
    items = [it.strip() for it in items if it.strip()]
    if not items:
        return answer.strip()
    if len(items) >= 6:
        keep = items[:6]
    elif len(items) >= 5:
        keep = items[:5]
    elif len(items) >= 3:
        keep = items[:3]
    else:
        keep = items
    return "\n".join(keep).strip()


# ------------------------------------------------------------
# 6) FAISS 헬퍼 및 단일 인덱스 구축 함수
# ------------------------------------------------------------
def faiss_from_texts(texts, embedding_model, metadatas=None):
    """ LangChain 버전에 따라 인자명이 다른 이슈를 try/except로 양쪽 지원 """
    try:
        return FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    except TypeError:
        return FAISS.from_texts(texts, embeddings=embedding_model, metadatas=metadatas)

def faiss_load_local(path, embedding_model):
    """ 저장된 로컬 인덱스를 안전하게 로드(역직렬화 허용) """
    try:
        return FAISS.load_local(path, embedding=embedding_model, allow_dangerous_deserialization=True)
    except TypeError:
        return FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def _index_file(path: str) -> str:
    return os.path.join(path, "index.faiss")

def _latest_json_mtime(folder: str) -> float:
    """ 폴더 내 JSON 파일들의 마지막 수정시각 중 최댓값 """
    times = []
    if not os.path.isdir(folder):
        return 0.0
    for f in os.listdir(folder):
        if f.lower().endswith(".json"):
            times.append(os.path.getmtime(os.path.join(folder, f)))
    return max(times) if times else 0.0

def _needs_rebuild(index_path: str, source_folder: str) -> bool:
    """ 인덱스가 없거나, 데이터가 더 최신이면 재빌드 필요 """
    if FORCE_REBUILD:
        return True
    idx = _index_file(index_path)
    if not os.path.exists(idx):
        return True
    return os.path.getmtime(idx) < _latest_json_mtime(source_folder)

def build_or_load_unified_disease_db():
    """
    통합 질병 인덱스 생성/로딩:
    - 검색 정확도를 위해 '증상'과 '증상.supplement' 가중치 반영
    - LLM 프롬프트에는 'clean_text'(가중치 없는 클린 버전) 사용
    """
    if _needs_rebuild(UNIFIED_DB_PATH, JSON_FOLDER):
        print("[Rebuild] 통합 질병 인덱스 (검색용/LLM용 분리)를 새로 생성합니다.")
        texts_for_embedding, metas = [], []
        files = sorted([f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")])

        for filename in files:
            with open(os.path.join(JSON_FOLDER, filename), encoding="utf-8") as f:
                data = json.load(f)

            disease_name = (data.get("병명") or "").strip()
            if not disease_name:
                continue

            symptom_data = data.get("증상", {})
            symptom_text = extract_any(symptom_data)

            # 증상.supplement를 별도로 추출(자유자연어 표현 보강)
            supplement_text = ""
            if isinstance(symptom_data, dict) and 'supplement' in symptom_data:
                supplement_text = extract_any(symptom_data.get('supplement'))

            # (검색용) 증상/보충을 3배 가중치로 확장하여 포함
            weighted_symptom_part = (f"[증상] {symptom_text}\n" + f"[증상.supplement] {supplement_text}\n") * 3

            # 나머지 필드(정의/원인/진단/치료 등)
            other_info_parts = [f"[병명] {disease_name}"]
            for key, value in data.items():
                if key not in ["병명", "증상"]:
                    content = extract_any(value)
                    if content:
                        other_info_parts.append(f"[{key}] {content}")
            other_info_part = "\n".join(other_info_parts)

            # (검색용) 가중치가 적용된 전체 텍스트
            weighted_document_text = (weighted_symptom_part + other_info_part).strip()
            texts_for_embedding.append(weighted_document_text)

            # (LLM용) 가중치 없는 클린 텍스트를 메타데이터에 보관
            clean_symptom_part = f"[증상] {symptom_text}"
            clean_document_text = (clean_symptom_part + "\n" + other_info_part).strip()
            metas.append({"병명": disease_name, "파일": filename, "clean_text": clean_document_text})

        if not texts_for_embedding:
            raise RuntimeError("통합 인덱스를 만들 텍스트가 없습니다.")

        db = faiss_from_texts(texts_for_embedding, embedding_model, metadatas=metas)
        db.save_local(UNIFIED_DB_PATH)
        return db
    else:
        print("[Load] 기존 통합 질병 인덱스를 불러옵니다.")
        return faiss_load_local(UNIFIED_DB_PATH, embedding_model)


# ------------------------------------------------------------
# 7) 단일 검색 함수
# ------------------------------------------------------------
def search_unified_db_with_scores(db, user_query: str, k: int) -> List[Tuple[any, float]]:
    """
    질의어를 받아 상위 k개 문서와 거리 점수를 함께 반환.
    - E5 모델을 고려하여 'query: ' 접두사를 붙임(다른 모델이어도 큰 문제 없음)
    - FAISS는 거리(작을수록 유사), 우리는 아래에서 간단히 유사도 점수로 변환해 사용.
    """
    q_fmt = f"query: {user_query}"
    if not db:
        return []
    return db.similarity_search_with_score(q_fmt, k)


# ------------------------------------------------------------
# 8) 프롬프트(SYSTEM) 및 메인 루프
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
    # 1) 인덱스 준비
    disease_db = build_or_load_unified_disease_db()
    print("\n✅ 통합 인덱스 준비 완료")

    # 2) 대화 루프
    while True:
        user_input = input("\n🩺 증상을 입력하세요 (종료: exit): ").strip()
        if user_input.lower() in ["exit", "종료", "quit"]:
            print("챗봇을 종료합니다.")
            break

        # 2-1) 검색 수행
        docs_with_scores = search_unified_db_with_scores(disease_db, user_input, k=K_DISEASE)

        # 2-2) 검색 실패 시: 일반 어시스턴트 응답
        if not docs_with_scores:
            print("[Info] 관련 질병 정보를 찾을 수 없습니다. 일반적인 답변을 시도합니다.")
            general_messages = [
                {"role": "system", "content": "당신은 사용자에게 친절하게 답변하는 AI 어시스턴트입니다."},
                {"role": "user", "content": user_input},
            ]
            try:
                answer = chat_with_friendli(general_messages)
                print(f"\n🧾 [EXAONE 32B 답변]\n{answer}")
            except Exception as e:
                print(f"[오류] API 호출 실패: {e}")
            continue

        # 2-3) 거리 → 간이 유사도(1/(1+dist))로 변환
        docs_with_sim_scores = [(doc, 1 / (1 + score)) for doc, score in docs_with_scores]
        unique_docs = docs_with_sim_scores

        # (출력 간소화로 1차 검색 결과/점수 미리보기는 비노출)
        # print("\n--- [1차 검색 결과 (Top 3)] ---")
        # ...

        # 2-4) 상위 1개 유사도 점수로 라우팅 판단
        top1_doc, top1_score = unique_docs[0]
        final_docs = []

        # (A) 비의료/잡담 라우팅
        if top1_score < LOW_CONF_THRESHOLD:
            print(f"[판단] 비의료 질문 (유사도: {top1_score:.2f} < {LOW_CONF_THRESHOLD})")
            general_messages = [
                {"role": "system", "content": "당신은 사용자에게 친절하게 답변하는 AI 어시스턴트입니다."},
                {"role": "user", "content": user_input},
            ]
            try:
                answer = chat_with_friendli(general_messages)
                print(f"\n🧾 [EXAONE 32B 답변]\n{answer}")
            except Exception as e:
                print(f"[오류] API 호출 실패: {e}")
            continue

        # (B) 확신도 판단
        is_confident = (
            top1_score >= HIGH_CONF_THRESHOLD
            and (len(unique_docs) < 3 or (unique_docs[0][1] - unique_docs[2][1]) >= SCORE_DIFF_THRESHOLD)
        )

        if is_confident:
            print(f"[판단] 확신도 높음 (유사도: {top1_score:.2f})")
            final_docs = [doc for doc, score in unique_docs[:MAX_DISEASES]]
        else:
            # (C) 확신도 낮음 → 추가 증상 요청 후 재검색
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

        # 2-5) LLM 생성: 검색 컨텍스트(클린 버전) + 사용자 질문
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

            # (출력 간소화로 LLM 컨텍스트 미리보기는 비노출)
            try:
                answer = chat_with_friendli(messages)
                print("\n🧾 [EXAONE 32B 최종 답변]")
                print(extract_numbered_block(answer))
            except Exception as e:
                print(f"[오류] 최종 답변 생성 실패: {e}")
