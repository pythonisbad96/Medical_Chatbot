# 상비약 추천 추가
# 영어 단어가 중간 중간에 들어감

import os
import ast
import re
import uuid
import json
import math  # NEW
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from openai import OpenAI


# 환경변수에서 EXAONE 32B용 API 정보 설정
OPENAI_BASE_URL = os.getenv(
    "FRIENDLI_BASE_URL", "https://api.friendli.ai/serverless/v1"
)
OPENAI_API_KEY = os.getenv("FRIENDLI_TOKEN", "")  # EXAONE API 키
MODEL_NAME = os.getenv("EXAONE_MODEL_NAME", "LGAI-EXAONE/EXAONE-4.0.1-32B")

# 데이터 폴더/경로
data = "./dataset"

# 1. CSV 로드
# 의료 데이터 여러 파일 불러오기
df_1200 = pd.read_csv(os.path.join(data, "1200_v1.csv"))
df_amc = pd.read_csv(os.path.join(data, "amc.csv"))
df_daily = pd.read_csv(os.path.join(data, "daily_dataset.csv"))
df_final = pd.read_csv(os.path.join(data, "final_v7.csv"))
df_kdca = pd.read_csv(os.path.join(data, "kdca.csv"))
df_snu = pd.read_csv(os.path.join(data, "snu.csv"))
df_mfds = pd.read_csv(os.path.join(data, "mfds_crawled_utf8.csv"))

# 임베딩 모델 준비(최초 1회)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")


# 텍스트를 단어 단위로 chunk로 분리
def chunk_by_words(texts, chunk_size=60):
    chunks = []
    for text in texts:
        if pd.isnull(text):
            continue
        words = str(text).split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    return chunks


# 2. 리스트 컬럼 안전 파싱 유틸
def _safe_list(x):
    if pd.isnull(x):
        return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            return [str(t).strip() for t in v if str(t).strip()]
        except Exception:
            pass
    # 콤마/슬래시/세미콜론/파이프 등 구분 기호 처리
    return [p.strip() for p in re.split(r"[,\|/;]", s) if p.strip()]


# 각 데이터셋을 FAISS 벡터DB로 저장(최초 1회)
def prepare_faiss():
    os.makedirs("vector_db", exist_ok=True)

    # 1200_v1
    texts_1200 = df_1200["label"].astype(str) + "\n" + df_1200["text"].astype(str)
    chunks_1200 = chunk_by_words(texts_1200.tolist(), chunk_size=60)
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
    chunks_amc = chunk_by_words(amc_texts.tolist(), chunk_size=60)
    db_amc = FAISS.from_texts(chunks_amc, embedding=embedding_model)
    db_amc.save_local("vector_db/faiss_db_amc")

    # daily
    daily_texts = df_daily["증상"].astype(str) + "\n" + df_daily["일상말"].astype(str)
    chunks_daily = chunk_by_words(daily_texts.tolist(), chunk_size=60)
    db_daily = FAISS.from_texts(chunks_daily, embedding=embedding_model)
    db_daily.save_local("vector_db/faiss_db_daily_dataset")

    # final_v7
    final_texts = df_final["label"].astype(str) + "\n" + df_final["text"].astype(str)
    chunks_final = chunk_by_words(final_texts.tolist(), chunk_size=60)
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
    chunks_kdca = chunk_by_words(kdca_texts.tolist(), chunk_size=60)
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
    chunks_snu = chunk_by_words(snu_texts.tolist(), chunk_size=60)
    db_snu = FAISS.from_texts(chunks_snu, embedding=embedding_model)
    db_snu.save_local("vector_db/faiss_db_snu")

    # 3. mfds (식약처 의약품/의약외품)
    mfds_ok = df_mfds.copy()
    if "status" in mfds_ok.columns:
        mfds_ok = mfds_ok[mfds_ok["status"].astype(str).str.lower().eq("ok")]

    prod = mfds_ok.get("제품명", "").astype(str)
    link = mfds_ok.get("상세링크", "").astype(str)
    ing_k = mfds_ok.get("성분목록", "")
    ing_e = mfds_ok.get("성분영문목록", "")
    eff = mfds_ok.get("효능효과", "").astype(str)
    dose = mfds_ok.get("용법용량", "").astype(str)

    mfds_texts = []
    for i in range(len(mfds_ok)):
        k_list = _safe_list(ing_k.iloc[i] if len(mfds_ok) > 0 else "")
        e_list = _safe_list(ing_e.iloc[i] if len(mfds_ok) > 0 else "")
        text = (
            f"제품명: {prod.iloc[i]}\n"
            f"성분(국문): {', '.join(k_list) if k_list else ''}\n"
            f"성분(영문): {', '.join(e_list) if e_list else ''}\n"
            f"효능효과: {eff.iloc[i]}\n"
            f"용법용량: {dose.iloc[i]}\n"
            f"출처: {link.iloc[i]}"
        )
        mfds_texts.append(text)

    chunks_mfds = chunk_by_words(mfds_texts, chunk_size=60)
    db_mfds = FAISS.from_texts(chunks_mfds, embedding=embedding_model)
    db_mfds.save_local("vector_db/faiss_db_mfds")


# 최초 1회만 실행 (필수 인덱스 존재 확인)
required_indexes = [
    "vector_db/faiss_db_1200_v1/index.faiss",
    "vector_db/faiss_db_amc/index.faiss",
    "vector_db/faiss_db_daily_dataset/index.faiss",
    "vector_db/faiss_db_final_v7/index.faiss",
    "vector_db/faiss_db_kdca/index.faiss",
    "vector_db/faiss_db_snu/index.faiss",
    "vector_db/faiss_db_mfds/index.faiss",
]
if not all(os.path.exists(p) for p in required_indexes):
    prepare_faiss()

# 벡터DB 경로 리스트 및 로드
db_paths = [
    "vector_db/faiss_db_1200_v1",
    "vector_db/faiss_db_amc",
    "vector_db/faiss_db_daily_dataset",
    "vector_db/faiss_db_final_v7",
    "vector_db/faiss_db_kdca",
    "vector_db/faiss_db_snu",
    "vector_db/faiss_db_mfds",
]
db_list = [
    FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    for db_path in db_paths
]

# 경로-DB 매핑 (검색 분기에서 사용)
path_db_pairs = list(zip(db_paths, db_list))


# 대화 세션 관리용 클래스 (대화내역/history 저장)
class ChatSession:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.history = []

    def save(self, path="session_logs/"):
        os.makedirs(path, exist_ok=True)
        with open(
            os.path.join(path, f"{self.session_id}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def append(self, user, message):
        self.history.append({"role": user, "message": message})


# LangChain 메모리(틀 유지용)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# LLM 프롬프트 생성
def make_prompt(user_question, retrieved_context):
    return f"""
당신은 의료 상담 챗봇입니다.

모든 답변은 반드시 존댓말(-입니다, -합니다, -해야 합니다 등)로만 작성하세요.
아래 [증상 정보]를 참고하여 요구된 항목만 출력하세요.

질문이 의료 관련이면 [증상 정보] 참고 1~5번, 무관하면 간결하게만 답변.
---
질문: {user_question}
---
[증상 정보]
{retrieved_context}
---
출력 예시:
1. 예상되는 병명 (각 병명마다 한 줄 설명 포함, 반드시 존댓말):
    - 첫 번째 병명: (설명)
    - 두 번째 병명: (설명)
    - 세 번째 병명: (설명, 선택적으로 포함)
2. 주요 원인:
3. 추천 진료과 (2~3과):
4. 예방 및 관리 방법:
5. 생활 시 주의사항:

(비의료 질문일 경우)
답변:
""".strip()


def make_medicine_prompt(user_question, retrieved_context):
    return f"""
당신은 의료 상담 챗봇입니다.

모든 답변은 반드시 존댓말(-입니다, -합니다, -해야 합니다 등)로만 작성하세요.
아래 [증상 정보]를 참고하여 **상비약 정보만** 출력하세요.
**최소 3개, 최대 5개**의 서로 다른 제품을 추천합니다. 각 제품은 아래 형식의 블록으로 **연속**해서 출력하고,
제품 간에는 **빈 줄 1줄**만 두세요. 표는 사용하지 않습니다. 동일/유사 성분의 중복 제품은 피하세요.  # CHANGED

제품: (제품명)
사용 방법: (사용 빈도/용량/부위 등 구체적으로)
주의사항: (금기/부작용/병용주의 등)
종합적인 관리: (보습/생활습관/악화요인 회피 등)
그외에 권고: (의료기관 방문 기준, 추가 검토 사항 등)

---
질문: {user_question}
---
[증상 정보]
{retrieved_context}
---
출력 예시:
제품: 하이드로코르티손 1% 크림
사용 방법: 하루 1~2회 가려운 부위에 얇게 바르십시오.
주의사항: 얼굴/접히는 부위 장기 사용은 피하시고 악화 시 사용을 중단하십시오.
종합적인 관리: 미지근한 물로 세안 후 바로 보습제를 충분히 바르십시오.
그외에 권고: 증상이 1주 이상 지속되거나 번지면 피부과 진료를 권합니다.

제품: 클로르헥시딘 가글액 0.12%
사용 방법: 하루 2회 15ml를 30초간 가글 후 뱉어내십시오.
주의사항: 삼키지 마시고, 사용 직후 음식/음료 섭취는 30분간 피하십시오.
종합적인 관리: 자극적인 음식은 피하고 수분 섭취를 늘리십시오.
그외에 권고: 출혈/부종이 심해지면 치과/이비인후과 진료를 권합니다.
""".strip()


# 존댓말 자동 후처리 함수(간단 버전)
def make_respectful(text):
    text = re.sub(r"([^.가-힣\w])?(\w+)[ ]*다([^\w가-힣])", r"\1\2니다\3", text)
    text = re.sub(r"임\.?", "입니다.", text)
    text = re.sub(r"임니다", "입니다", text)
    return text


# 2~5번 항목을 여러 줄로 변환하는 후처리 함수
def format_multiline_sections(text):
    section_pat = re.compile(r"([2-5]\..*?:)(.*?)(?=\n[3-5]\.|$)", re.DOTALL)

    def _split_lines(match):
        title = match.group(1)
        content = match.group(2)
        # 콤마, ·, ㆍ 등으로 분리
        lines = []
        for item in re.split(r",|·|ㆍ", content):
            item = item.strip(" \n-")
            if not item:
                continue
            if not item.startswith("-"):
                lines.append(f"    - {item}")
            else:
                lines.append(f"    {item}")
        return title + "\n" + "\n".join(lines) + "\n"

    return section_pat.sub(_split_lines, text)


# OpenAI(AX4 등) 클라이언트 준비
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# 대화 세션 시작
session = ChatSession()
print(f"새로운 세션이 시작되었습니다. session_id: {session.session_id}\n")

while True:
    # 사용자 질문 입력
    user_question = input("질문을 입력하세요(종료: exit): ")
    if user_question.lower() in ["exit", "quit", "종료"]:
        print("프로그램을 종료합니다.")
        session.save()
        break

    # 최근 2턴까지의 유저질문을 합쳐서 RAG 검색 쿼리 생성
    prev_utts = [h["message"] for h in session.history if h["role"] == "user"]
    rag_query = (
        " ".join(prev_utts[-2:] + [user_question]) if prev_utts else user_question
    )

    # ------ 전체 DB 검색 + MFDS 지배도 라우팅 (키워드 0개) ------
    top_k_per_db_default = 3
    all_scored = []  # [(path, doc, score)]
    mfds_scored = []
    other_scored = []

    for path, db in path_db_pairs:
        # 상비약 다양성 확보를 위해 MFDS는 더 많이 가져옴  # CHANGED
        k = 12 if path.endswith("faiss_db_mfds") else top_k_per_db_default
        results = db.similarity_search_with_score(rag_query, k=k)
        for d, s in results:
            tup = (path, d, s)
            all_scored.append(tup)
            if path.endswith("faiss_db_mfds"):
                mfds_scored.append(tup)
            else:
                other_scored.append(tup)

    # 전역 상위 랭크 뽑기
    all_scored_sorted = sorted(
        all_scored, key=lambda x: x[2]
    )  # score ASC (더 낮을수록 가깝다고 가정)
    topM = all_scored_sorted[:8]  # 지배도 평가용 상위 8개

    # 지배도 판정 규칙
    mfds_min = min([s for (p, _, s) in mfds_scored], default=math.inf)
    other_min = min([s for (p, _, s) in other_scored], default=math.inf)
    mfds_hits_in_topM = sum(1 for (p, _, _) in topM if p.endswith("faiss_db_mfds"))
    MARGIN = 0.30  # 필요시 조정

    medicine_intent = (mfds_hits_in_topM >= 3) or (mfds_min + MARGIN <= other_min)

    # 컨텍스트 구성
    seen = set()
    context_candidates = []

    def _push_from_scored(scored_list, limit):  # CHANGED: limit 파라미터
        for _, doc, _ in scored_list:
            text = (doc.page_content or "").strip()
            if len(text) < 30:
                continue
            if text in seen:
                continue
            context_candidates.append(text)
            seen.add(text)
            if len(context_candidates) == limit:
                break

    # 상비약 모드면 컨텍스트를 최대 5개까지 확보해 다양성 강화
    # CHANGED
    context_limit = 5 if medicine_intent else 3

    if medicine_intent:
        mfds_sorted = sorted(mfds_scored, key=lambda x: x[2])
        _push_from_scored(mfds_sorted, context_limit)
    else:
        _push_from_scored(all_scored_sorted, context_limit)

    retrieved_context = (
        "\n---\n".join(context_candidates)[:1600]
        if context_candidates
        else ""  # CHANGED: 약간 여유
    )

    # 대화내역에 user질문 추가, 최신 4턴만 유지(2쌍)
    session.append("user", user_question)
    session.history = session.history[-4:]

    # LangChain 메모리(대화 이력 유지)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for h in session.history:
        if h["role"] == "user":
            memory.chat_memory.add_user_message(h["message"])
        else:
            memory.chat_memory.add_ai_message(h["message"])

    # 프롬프트 선택 (컨텍스트 키워드 검사 제거)
    if medicine_intent:
        dynamic_prompt = make_medicine_prompt(user_question, retrieved_context)
    else:
        dynamic_prompt = make_prompt(user_question, retrieved_context)

    messages = []
    messages.append({"role": "system", "content": dynamic_prompt})
    for m in memory.chat_memory.messages:
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": user_question})

    print("\n[EXAONE_32B 응답]\n", end="", flush=True)  # 응답 시작 줄바꿈

    # OpenAI LLM 스트리밍 호출
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )
    answer_chunks = []
    for chunk in stream:
        delta = getattr(chunk.choices[0].delta, "content", None)
        if delta:
            answer_chunks.append(delta)
            print(delta, end="", flush=True)
    print()
    answer = "".join(answer_chunks).strip()

    # 후처리: 상비약 모드/일반 모드 분리
    if medicine_intent:
        # 상비약 전용 출력은 그대로 사용 (프롬프트가 3~5개 블록을 강제)  # CHANGED
        answer_only = make_respectful(answer)
    else:
        # 1~5번 항목만 추출
        matches = re.findall(
            r"1\..*?\n2\..*?\n3\..*?\n4\..*?\n5\..*(?=\n|$)", answer, flags=re.DOTALL
        )
        answer_only = matches[-1].strip() if matches else answer

        # 1번 항목 줄바꿈 정리(병명-설명 여러 줄)
        pattern = r"(1\..*?)(\n2\..*)"
        m = re.search(pattern, answer_only, flags=re.DOTALL)
        if m:
            section1 = m.group(1)
            section2plus = m.group(2)
            lines = section1.split("\n")
            bullet_lines = [
                line.rstrip() for line in lines if line.strip().startswith("- ")
            ]
            if bullet_lines:
                merged_bullets = "\n    ".join(bullet_lines)
                section1_fixed = lines[0] + "\n    " + merged_bullets
                answer_only = section1_fixed + section2plus

        # 항상 존댓말 후처리 + 다중라인
        answer_only = make_respectful(answer_only)
        answer_only = format_multiline_sections(answer_only)

    # 대화내역에 assistant 답변 추가
    session.append("assistant", answer_only)
