# -*- coding: utf-8 -*-
"""
medical_main.py (최종본)

[무엇을 하는 파일인가요?]
- 콘솔에서 증상을 입력받아 아래 파이프라인으로 답변을 생성합니다.
  1) 의료성 판별(간단 RAG 평균 유사도)
  2) 질병/약품 검색(증상 전용 인덱스 + E5 접두사 + similarity 또는 MMR)
  3) Evidence Gating(근거가 약하면 안전 모드 전환)
  4) A.X 4.0(OpenAI 호환) 호출로 최종 답변 생성

[핵심 리트리벌 강화 포인트]
- E5 접두사(query:/passage:) 적용
- 증상 전용 인덱스(증상/관련_증상_표현만 문장 단위) 별도 구축
- similarity 우선 + MMR 토글 가능(환경변수)
- 구어체 → 의학어 간단 정규화(CANON + JSON 사전)
- Evidence Gating(평균/최대 유사도, 문서 수 임계)

[평가 스크립트 호환성]
- 외부 평가 스크립트가 이 모듈을 import 하여 아래 함수/상수들을 그대로 사용합니다.
  (build_or_load_*, search_*, is_medical_by_rag, evidence_score/evidence_ok,
   get_disease_from_doc, dedupe_by_disease, embedding_model, SYSTEM_PROMPT/SAFE_PROMPT 등)

[필수 환경변수]
- ADOTX_API_KEY : A.X 4.0 API 키 (없으면 즉시 오류)

[필요 패키지]
- langchain_huggingface, langchain_community, openai, faiss, pandas, numpy, rapidfuzz, python-dotenv(옵션)
"""

# -------------------------
# 표준/서드파티 모듈 import
# -------------------------
import os                       # 경로/환경변수 처리
import json                     # JSON 로드
import re                       # 정규표현식(텍스트 가공)
import numpy as np              # 수치연산(유사도 계산 등)
import pandas as pd             # CSV 로드(상비약)
from typing import List
from rapidfuzz import fuzz      # (현재 파일에서는 사용 빈도 낮지만 추후 유용)

# ---- (선택) .env 지원: 로컬 개발 시 편하게 환경변수 주입 ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---- GPU 자동 감지: torch가 있으면 CUDA 사용, 없으면 CPU ----
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ---- LangChain: 임베딩 + FAISS 벡터DB ----
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---- A.X 4.0(OpenAI 호환 API) ----
from openai import OpenAI

# -------------------------
# 경로/DB 디렉토리 등 기본 설정
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # medical_main.py 위치 (src/)

# 데이터 폴더 (루트 기준 ../data)
JSON_FOLDER   = os.path.join(BASE_DIR, "../data/json_diseases_augmented")  # 질병 JSON 폴더
MED_CSV_PATH  = os.path.join(BASE_DIR, "../data/mfds_crawled_utf8.csv")   # 상비약 CSV
COLLOQUIAL_JSON = os.path.join(BASE_DIR, "../data/colloquial_to_disease.json")

# 벡터 DB 폴더 (루트 기준 ../vector_e5_large_with_meds)
DB_DIR        = os.path.join(BASE_DIR, "../vector_e5_large_with_meds")
DISEASE_DB_PATH = os.path.join(DB_DIR, "faiss_disease_db")
SYM_DB_PATH     = os.path.join(DB_DIR, "faiss_disease_db_symptom_only")
MED_DB_PATH     = os.path.join(DB_DIR, "faiss_med_db")
MEDICALNESS_DB  = os.path.join(DB_DIR, "faiss_medicalness")

os.makedirs(DB_DIR, exist_ok=True)              # 없으면 생성

# -------------------------
# 런타임 플래그(환경변수로 토글)
# -------------------------
USE_SYMPTOM_ONLY = os.getenv("USE_SYMPTOM_ONLY", "1") == "1"  # 증상 전용 인덱스 사용(기본 ON)
USE_E5_PREFIX    = os.getenv("USE_E5_PREFIX", "1") == "1"     # E5 접두사 사용(기본 ON)
USE_MMR          = os.getenv("USE_MMR", "0") == "1"           # MMR 토글(기본 OFF)
LAMBDA_MULT      = float(os.getenv("LAMBDA_MULT", "0.2"))     # MMR 다양성 가중
FETCH_MULT       = int(os.getenv("FETCH_MULT", "5"))          # MMR fetch_k = k*FETCH_MULT

# -------------------------
# 임베딩 모델(E5-large) 로드
# -------------------------
# normalize_embeddings=True → 내적(dot) == 코사인 유사도
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": DEVICE},                      # ✅ GPU 자동감지 반영
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
)

# -------------------------
# 검색/컨텍스트 파라미터
# -------------------------
K_DISEASE    = 10    # 질병 검색 Top-k
K_MED        = 5     # 상비약 검색 Top-k
MAX_DISEASES = 5     # 컨텍스트에 넣을 서로 다른 병명 수(중복 제거 후)
MAX_MEDS     = 3     # 컨텍스트에 넣을 상비약 수
CTX_CHARS    = 2000  # LLM에 보낼 컨텍스트 최대 길이
MED_TOPK     = 5     # 의료성 판별용 검색 k
MED_TH       = 0.55  # 의료성 평균 유사도 임계값

# -------------------------
# A.X 4.0 클라이언트 설정 (키 필수)
# -------------------------
AX_BASE_URL = "https://guest-api.sktax.chat/v1"
AX_API_KEY  = os.getenv("ADOTX_API_KEY", "").strip()  # ✅ 기본값 제거: 반드시 환경변수로 주입
AX_MODEL    = "ax4"

if not AX_API_KEY:
    raise RuntimeError("환경변수 ADOTX_API_KEY가 설정되어 있지 않습니다. A.X 4.0 API 키를 넣어주세요.")

client = OpenAI(base_url=AX_BASE_URL, api_key=AX_API_KEY)

def chat_with_ax(messages, **gen_opts):
    """
    A.X 4.0(chat.completions) 호출 유틸 함수
    - messages: [{"role":"system/user/assistant", "content":"..."}]
    - gen_opts: temperature, max_tokens 등(옵션)
    """
    completion = client.chat.completions.create(
        model=AX_MODEL,
        messages=messages,
        temperature=gen_opts.get("temperature", 0.3),
        max_tokens=gen_opts.get("max_tokens", 1024),
    )
    return completion.choices[0].message.content.strip()

# -------------------------
# 구어체 → 의학어 정규화 사전(JSON)
# -------------------------
try:
    with open(COLLOQUIAL_JSON, "r", encoding="utf-8") as f:
        COLLOQUIAL_MAP = json.load(f)
except FileNotFoundError:
    COLLOQUIAL_MAP = {}

def normalize_query_dict(q: str) -> str:
    """
    JSON 사전 기반 정규화
    - 사전 값이 list/tuple로 되어 있어도 ensure_text로 문자열화해서 반환
    """
    key = ensure_text(q).strip()
    val = COLLOQUIAL_MAP.get(key, key)
    return ensure_text(val)

# -------------------------
# 간단 치환(CANON): 구어체 패턴을 의학 용어로 교체
# -------------------------
CANON = {
    "쌕쌕거려요":"천명", "쌕쌕거림":"천명", "쌕쌕":"천명",
    "가슴이 답답":"흉부 압박감", "가슴답답":"흉부 압박감",
    "숨이 가빠":"호흡곤란", "숨이차요":"호흡곤란", "숨이차":"호흡곤란",
    "토할 것 같":"오심", "속이 울렁":"오심", "메스꺼":"오심",
    "토했":"구토", "구역질":"구토",
    "두근두근":"심계항진", "심장이 빨리 뛰":"심계항진",
    "어지러워요":"현훈", "빙글빙글":"현훈", "눈앞이 핑":"현훈",
    "콧물나":"비루", "코막혀":"비폐색", "코막힘":"비폐색",
    "목이 아파":"인두통", "목따가움":"인두통",
    "가래":"객담", "피가래":"혈담",
    "열이 나":"발열", "미열":"발열",
    "오줌눌때 아파":"배뇨통", "소변볼때 아파":"배뇨통",
    "자주 소변":"빈뇨", "밤에 자주 소변":"야뇨",
    "배아파요":"복통", "속이 아파":"복통",
    "설사해요":"설사", "묽은 변":"설사", "피변":"혈변",
    "속쓰려요":"가슴쓰림", "쓰림":"가슴쓰림",
    "기침나와요":"기침", "기침 심해요":"기침",
    "피부가 가려":"소양증", "가렵":"소양증",
    "부었다":"부종", "붓기":"부종",
}
def normalize_query_colloquial(q: str) -> str:
    """구어체 표현을 의학어로 간단 치환(입력은 무엇이 와도 문자열로 정규화)"""
    s = ensure_text(q)
    for k, v in CANON.items():
        s = s.replace(k, v)
    return s


# -------------------------
# 텍스트 전처리 유틸
# -------------------------

# (파일 상단 유틸 함수들 근처에 추가)
def ensure_text(x) -> str:
    """
    어떤 타입이 와도 안전하게 문자열로 변환합니다.
    - list/tuple/set: 공백으로 합쳐 하나의 문자열로
    - 그 외: str() 시도, 실패 시 빈 문자열
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple, set)):
        return " ".join(map(str, x))
    try:
        return str(x)
    except Exception:
        return ""


def extract_text(field_dict):
    """
    JSON 섹션(dict)에서 문자열만 깔끔히 추출
    - 값이 list면 줄바꿈으로 합칩니다.
    - "None"/빈값은 제거합니다.
    """
    if not isinstance(field_dict, dict):
        return ""
    texts = []
    for v in field_dict.values():
        if isinstance(v, list):
            vals = [str(x) for x in v if x not in (None, "", "None")]
            if vals:
                texts.append("\n".join(vals))
        elif isinstance(v, str):
            if v and v != "None":
                texts.append(v)
        elif v is not None:
            texts.append(str(v))
    return "\n".join(texts).strip()

def chunk_text(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    """
    단순 슬라이딩 청킹: 길이가 size를 넘으면 overlap만큼 겹치며 분할
    """
    text = text or ""
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        out.append(text[i:j])
        if j >= n: break
        i = max(0, j - overlap)
    return out

def split_sentences(text: str) -> List[str]:
    """
    간단 문장 분리(영문 문장부호 기준, 한국어도 대체로 동작)
    """
    if not text:
        return []
    s = re.sub(r"\s+", " ", text).strip()
    return [x for x in re.split(r"(?<=[\.?!])\s+", s) if x]

# -------------------------
# 문서에서 병명 추출 + 중복 제거
# -------------------------
def get_disease_from_doc(doc):
    """
    벡터DB 문서 메타데이터/본문에서 병명을 추출합니다.
    - 메타데이터 '병명' 우선, 없으면 본문에서 [병명] 라인을 파싱합니다.
    """
    if getattr(doc, "metadata", None) and "병명" in doc.metadata:
        return doc.metadata["병명"]
    m = re.search(r"\[병명\]\s*(.*)", getattr(doc, "page_content", "") or "")
    return m.group(1).strip() if m else None

def dedupe_by_disease(docs, max_items: int) -> List:
    """
    동일 병명 중복 제거: 같은 병명은 한 번만 남깁니다.
    """
    seen, uniq = set(), []
    for d in docs:
        dis = get_disease_from_doc(d) or "_unknown_"
        if dis in seen:
            continue
        seen.add(dis)
        uniq.append(d)
        if len(uniq) >= max_items:
            break
    return uniq

# -------------------------
# E5 접두사 포맷터
# -------------------------
def format_query_for_e5(q: str) -> str:
    """E5 권장 포맷: 쿼리에 'query:' 접두사(토글 가능)"""
    return f"query: {q}" if USE_E5_PREFIX else q

# -------------------------
# 인덱스: 질병(전체), 질병(증상 전용), 상비약, 의료성
# -------------------------
def faiss_from_texts(texts, embedding_model, metadatas=None):
    """
    LangChain FAISS 버전 차이 대응 래퍼: embedding / embeddings 인자명을 모두 시도
    """
    try:
        return FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    except TypeError:
        return FAISS.from_texts(texts, embeddings=embedding_model, metadatas=metadatas)

def faiss_load_local(path, embedding_model):
    """
    LangChain FAISS 버전 차이 대응 래퍼: embedding / embeddings 인자명을 모두 시도
    """
    try:
        return FAISS.load_local(path, embedding=embedding_model, allow_dangerous_deserialization=True)
    except TypeError:
        return FAISS.load_local(path, embeddings=embedding_model, allow_dangerous_deserialization=True)

def build_or_load_faiss_disease():
    """
    (전체 섹션) 질병 인덱스: [병명] + [정의/원인/증상/진단/치료/관련_증상_표현]
    - 각 청크 앞에 'passage:' 접두사(E5 권장) 부여
    """
    idx = os.path.join(DISEASE_DB_PATH, "index.faiss")
    if os.path.exists(idx):
        return faiss_load_local(DISEASE_DB_PATH, embedding_model)

    texts, metas = [], []
    files = sorted([f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")])
    for filename in files:
        with open(os.path.join(JSON_FOLDER, filename), encoding="utf-8") as f:
            data = json.load(f)
        disease = (data.get("병명") or "").strip()
        sections = [
            ("정의",   extract_text(data.get("정의", {}))),
            ("원인",   extract_text(data.get("원인", {}))),
            ("증상",   extract_text(data.get("증상", {}))),
            ("진단",   extract_text(data.get("진단", {}))),
            ("치료",   extract_text(data.get("치료", {}))),
            ("관련_증상_표현", "\n".join(data.get("관련_증상_표현", []))),
        ]
        for sec_name, sec_text in sections:
            if not sec_text:
                continue
            for j, ch in enumerate(chunk_text(sec_text)):
                text = f"passage: [병명] {disease}\n[{sec_name}] {ch}".strip()
                texts.append(text)
                metas.append({"병명": disease, "파일": filename, "섹션": sec_name, "chunk_id": j})

    db = faiss_from_texts(texts, embedding_model, metadatas=metas)
    db.save_local(DISEASE_DB_PATH)
    return db

def build_or_load_faiss_disease_symptom_only():
    """
    (증상 전용) 질병 인덱스: [증상] + [관련_증상_표현]만 문장 단위로 저장
    - 정의/진단/치료 등 노이즈를 제거해 증상 질의에 민감
    - 각 문장 앞에 'passage:' 접두사 부여
    """
    idx = os.path.join(SYM_DB_PATH, "index.faiss")
    if os.path.exists(idx):
        return faiss_load_local(SYM_DB_PATH, embedding_model)

    texts, metas = [], []
    files = sorted([f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")])
    for filename in files:
        with open(os.path.join(JSON_FOLDER, filename), encoding="utf-8") as f:
            data = json.load(f)
        disease = (data.get("병명") or "").strip()

        raw_sym   = extract_text(data.get("증상", {}))
        raw_exprs = "\n".join(data.get("관련_증상_표현", []))
        merged = (raw_sym + ("\n" if raw_sym and raw_exprs else "") + raw_exprs).strip()
        if not merged:
            continue

        # 문장 단위 분리
        sents = re.split(r"(?<=[.!?])\s+|\n+", merged)
        for j, sent in enumerate([s for s in sents if s.strip()][:200]):
            text = f"passage: [병명] {disease}\n[증상] {sent.strip()}"
            texts.append(text)
            metas.append({"병명": disease, "파일": filename, "섹션": "증상/표현", "chunk_id": j})

    db = faiss_from_texts(texts, embedding_model, metadatas=metas)
    db.save_local(SYM_DB_PATH)
    return db

def build_or_load_faiss_meds():
    """
    상비약 인덱스: [제품명] + [효능효과]
    - 인코딩을 utf-8-sig 우선 시도하여 엑셀 CSV(BOM)도 안전하게 로드
    """
    idx = os.path.join(MED_DB_PATH, "index.faiss")
    if os.path.exists(idx):
        return faiss_load_local(MED_DB_PATH, embedding_model)

    # ✅ 인코딩 안전 처리
    try:
        df = pd.read_csv(MED_CSV_PATH, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(MED_CSV_PATH)

    texts, metas = [], []
    for _, row in df.iterrows():
        product_name = row.get("제품명", "")
        efficacy     = row.get("효능효과", "")
        if isinstance(product_name, str) and isinstance(efficacy, str) and efficacy.strip() and efficacy != "-":
            text = f"[제품명] {product_name}\n[효능효과] {efficacy}"
            texts.append(text)
            metas.append({"제품명": product_name, "상세링크": row.get("상세링크", "")})

    if not texts:
        raise ValueError("상비약 인덱스를 생성할 데이터가 없습니다. CSV를 확인하세요.")

    db = faiss_from_texts(texts, embedding_model, metadatas=metas)
    db.save_local(MED_DB_PATH)
    return db

def build_or_load_medicalness_index():
    """
    의료성 판별 소형 인덱스(정의/원인/증상/진단/치료/관련_증상_표현의 일부 문장)
    - 질문이 의료 관련인지 사전 필터링하는 용도
    """
    os.makedirs(MEDICALNESS_DB, exist_ok=True)
    idx_file = os.path.join(MEDICALNESS_DB, "index.faiss")
    if os.path.exists(idx_file):
        return faiss_load_local(MEDICALNESS_DB, embedding_model)

    samples, metas = [], []
    files = [fn for fn in os.listdir(JSON_FOLDER) if fn.endswith(".json")]
    for filename in files:
        with open(os.path.join(JSON_FOLDER, filename), encoding="utf-8") as f:
            data = json.load(f)
        for sec in ("정의", "원인", "증상", "진단", "치료", "관련_증상_표현"):
            # 섹션이 dict이면 extract_text, list이면 join
            txt = extract_text(data.get(sec, {})) if isinstance(data.get(sec), dict) else "\n".join(data.get(sec, []))
            for sent in split_sentences(txt)[:2]:  # 섹션당 최대 2문장 샘플
                if sent:
                    samples.append(sent)
                    metas.append({"파일": filename, "섹션": sec})

    if not samples:
        raise RuntimeError("의료성 인덱스 샘플이 없습니다. JSON 구조를 확인하세요.")

    db = faiss_from_texts(samples, embedding_model, metadatas=metas)
    db.save_local(MEDICALNESS_DB)
    return db

# -------------------------
# 의료성 판별 (LangChain 버전 호환)
# -------------------------
def is_medical_by_rag(text: str, retriever):
    """
    쿼리와 의료성 인덱스 Top-k 문장 간 임베딩 코사인 평균으로 의료성 판별
    - normalize_embeddings=True → dot == cosine
    - retriever.invoke() 없으면 get_relevant_documents()로 폴백(버전 호환)
    """
    if not text:
        return False

    # 쿼리 벡터 생성
    q_vec = np.array(embedding_model.embed_query(text), dtype=np.float32)

    # ✅ 버전 호환: invoke → 실패 시 get_relevant_documents
    try:
        docs = retriever.invoke(text)
    except Exception:
        docs = retriever.get_relevant_documents(text)

    if not docs:
        return False

    # 후보 문서 벡터와 평균 코사인 유사도
    cand_vecs = embedding_model.embed_documents([d.page_content for d in docs])
    sims = [float(np.dot(q_vec, np.array(v, dtype=np.float32))) for v in cand_vecs]
    avg_sim = float(np.mean(sims))
    print(f"[MED-RAG] sims={', '.join(f'{s:.3f}' for s in sims)} | avg={avg_sim:.3f}")
    return avg_sim >= MED_TH

# -------------------------
# Evidence Gating (근거 부족 시 안전 모드)
# -------------------------
EVIDENCE_MIN_AVG = float(os.getenv("EVIDENCE_MIN_AVG", "0.35"))  # 평균 유사도 하한(권장 시작점)
EVIDENCE_MIN_MAX = float(os.getenv("EVIDENCE_MIN_MAX", "0.45"))  # 최대 유사도 하한
EVIDENCE_MIN_N   = int(os.getenv("EVIDENCE_MIN_N",   "3"))       # 최소 문서 수

def evidence_score(docs, formatted_query: str):
    """
    검색 결과(docs)와 쿼리(formatted_query: query: 포함) 간의 근접도를 측정
    - 반환: (avg_sim, max_sim, n_docs)
    """
    if not docs:
        return 0.0, 0.0, 0
    q_vec = np.array(embedding_model.embed_query(formatted_query), dtype=np.float32)
    d_vecs = embedding_model.embed_documents([d.page_content for d in docs])
    sims = [float(np.dot(q_vec, np.array(v, dtype=np.float32))) for v in d_vecs]
    return float(np.mean(sims)), float(np.max(sims)), len(docs)

def evidence_ok(avg_sim, max_sim, n_docs) -> bool:
    """
    평균/최대 유사도와 문서 수가 임계 이상이면 True → 일반 모드
    아니면 False → 안전 모드
    """
    if n_docs < EVIDENCE_MIN_N:    return False
    if avg_sim < EVIDENCE_MIN_AVG: return False
    if max_sim < EVIDENCE_MIN_MAX: return False
    return True

# -------------------------
# 시스템 프롬프트(일반/안전)
# -------------------------
SYSTEM_PROMPT = """
당신은 의료 상담 챗봇입니다.

사용자 질문이 건강/증상/의학 관련이면, 아래 [질병 정보]와 [상비약 정보]를 참고하여 1~6번 항목을 작성하세요.
6번은 반드시 실제 상비약 제품명을 활용하세요. 없으면 '관련 증상에 맞는 상비약 정보가 없습니다'라고 답변하세요.

# 서식 금지 규칙(중요):
- 마크다운을 일절 사용하지 마세요. 굵게(**), 기울임(*, _), 헤더(#), 코드블록(```), 인라인코드(`) 모두 금지.
- 특히 별표(*) 문자는 출력하지 마세요. 목록은 '1. 2. 3.' 같은 숫자 리스트만 사용하세요.
- 평문(plain text)만 출력하세요.

📝 출력 형식:
1. 예상되는 병명 (2~3가지): (첫 번째 병명은 간단한 설명도 포함)
2. 주요 원인:
3. 추천 진료과 (2~3과):
4. 예방 및 관리 방법:
5. 생활 시 주의사항:
6. 상비약 추천(실제 제품):
""".strip()

SAFE_PROMPT = """
당신은 의료 안전 모드의 상담 챗봇입니다.

근거가 충분하지 않을 때에는
- 특정 병명 진단 또는 상비약 제품 추천을 하지 말고,
- 추가 정보 질문과, 응급 경고 신호/내원 기준/일반적인 생활관리만 안내하세요.
- 출력은 숫자 목록(1~)의 평문만 사용하세요.

📝 출력 형식:
1. 추가로 확인이 필요한 정보:
2. 응급실 또는 즉시 진료가 필요한 위험 신호:
3. 일시적인 자가 관리 방법(진단/약 추천 금지):
""".strip()

# -------------------------
# 검색 함수 (similarity 기본, 필요 시 MMR)
# -------------------------
def search_diseases(db, user_query: str, k: int = K_DISEASE):
    """
    1) 구어체→의학어 간단 치환
    2) E5 권장 포맷(query:) 적용
    3) similarity_search(기본) 또는 MMR(옵션)로 검색
    """
    q_norm = normalize_query_colloquial(user_query)     # CANON 치환
    q_fmt  = format_query_for_e5(q_norm)                # E5 query: 접두사
    if USE_MMR:
        fetch_k = max(k * FETCH_MULT, k)
        docs = db.max_marginal_relevance_search(
            q_fmt, k=k, fetch_k=fetch_k, lambda_mult=LAMBDA_MULT
        )
    else:
        docs = db.similarity_search(q_fmt, k=k)
    return docs, q_fmt

def search_meds(med_db, user_query: str, k: int = K_MED):
    """
    상비약은 키워드/유사도 매칭으로 충분한 경우가 많음 → similarity 사용
    (동일하게 CANON 치환 + E5 query 포맷)
    """
    q_norm = normalize_query_colloquial(user_query)
    q_fmt  = format_query_for_e5(q_norm)
    docs = med_db.similarity_search(q_fmt, k=k)
    return docs

# -------------------------
# 메인 루프(콘솔 인터페이스)
# -------------------------
if __name__ == "__main__":
    # 1) 인덱스 준비(증상 전용/전체 인덱스 모두 구축 후 플래그로 선택)
    disease_db_sym = build_or_load_faiss_disease_symptom_only()   # 증상 전용 인덱스
    disease_db_all = build_or_load_faiss_disease()                 # 전체 섹션 인덱스
    disease_db     = disease_db_sym if USE_SYMPTOM_ONLY else disease_db_all

    # ✅ 평가/디버깅 편의를 위해 모듈 전역으로도 노출
    globals()["disease_db"] = disease_db

    # 2) 상비약/의료성 인덱스 준비
    med_db         = build_or_load_faiss_meds()
    medicalness_db = build_or_load_medicalness_index()

    # 3) 의료성 판별 리트리버 (k=MED_TOPK)
    medicalness_retriever = medicalness_db.as_retriever(search_kwargs={"k": MED_TOPK})

    # 4) 콘솔 대화 루프 시작
    while True:
        user_input = input("\n🩺 증상을 입력하세요 (종료: exit): ").strip()
        if user_input.lower() in ["exit", "종료", "quit"]:
            print("챗봇을 종료합니다.")
            break

        # 4-1) 의료성 판별: 의료 관련이 아니면 일반 대화로 응답
        if is_medical_by_rag(user_input, medicalness_retriever):
            # 4-2) JSON 사전 기반 정규화 → CANON 치환 → E5 query 포맷
            normalized_input = normalize_query_dict(user_input)
            print(f"[정규화] {user_input}  →  {normalized_input}")

            # 4-3) 질병/약품 검색
            disease_docs, q_fmt = search_diseases(disease_db, normalized_input, k=K_DISEASE)
            med_docs            = search_meds(med_db, normalized_input, k=K_MED)

            # 4-4) Evidence Gating 점수 산출
            avg_sim, max_sim, n_docs = evidence_score(disease_docs, q_fmt)

            if evidence_ok(avg_sim, max_sim, n_docs):
                # 근거 충분 → 일반 모드 컨텍스트 구성(병명 기준 중복 제거)
                disease_docs_dedup = dedupe_by_disease(disease_docs, MAX_DISEASES)
                disease_context = "\n---\n".join([doc.page_content for doc in disease_docs_dedup])
                med_context     = "\n---\n".join([doc.page_content for doc in med_docs[:MAX_MEDS]])
                final_context   = f"[질병 정보]\n{disease_context}\n\n[상비약 정보]\n{med_context}"

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": final_context[:CTX_CHARS]},
                    {"role": "user",   "content": user_input},
                ]

                # (디버그) LLM 컨텍스트 미리보기
                print("\n🧩 [LLM 컨텍스트 미리보기]")
                print(final_context[:CTX_CHARS])
                print("-" * 60)

            else:
                # 근거 부족 → 안전 모드: 진단/약추천 대신 추가 질문/응급기준/생활관리
                safe_hint = (
                    "현재 검색 근거가 충분하지 않아 안전 모드로 전환합니다.\n"
                    f"- avg_sim={avg_sim:.3f}, max_sim={max_sim:.3f}, n_docs={n_docs}\n"
                    "진단/약추천 대신, 추가 확인이 필요한 정보와 응급 기준, 생활 관리만 안내합니다."
                )
                print("\n🧯 [Evidence Gating] " + safe_hint)
                messages = [
                    {"role": "system", "content": SAFE_PROMPT},
                    {"role": "user",   "content": user_input},
                ]

        else:
            # 비의료 질문: 일반 대화 프롬프트로 응답
            messages = [
                {"role": "system", "content": "당신은 친절한 AI 챗봇입니다. 무엇을 도와드릴까요?"},
                {"role": "user",   "content": user_input}
            ]

        # 4-5) A.X 4.0 호출 및 출력
        try:
            answer = chat_with_ax(messages)
            print("\n🧾 [A.X 4.0 응답 결과]")
            print(answer)
        except Exception as e:
            print(f"[오류] A.X 4.0 API 호출 실패: {e}")
