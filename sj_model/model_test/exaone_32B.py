import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from openai import OpenAI

import re
import uuid
import json

# 환경변수에서 EXAONE 32B용 API 정보 설정
OPENAI_BASE_URL = os.getenv(
    "FRIENDLI_BASE_URL", "https://api.friendli.ai/serverless/v1"
)
OPENAI_API_KEY = os.getenv(
    "FRIENDLI_TOKEN", ""
)  # EXAONE API 키
MODEL_NAME = os.getenv("EXAONE_MODEL_NAME", "LGAI-EXAONE/EXAONE-4.0.1-32B")

# 데이터 폴더/경로
data = "./dataset"

# 의료 데이터 여러 파일 불러오기
df_1200 = pd.read_csv(os.path.join(data, "1200_v1.csv"))
df_amc = pd.read_csv(os.path.join(data, "amc.csv"))
df_daily = pd.read_csv(os.path.join(data, "daily_dataset.csv"))
df_final = pd.read_csv(os.path.join(data, "final_v7.csv"))
df_kdca = pd.read_csv(os.path.join(data, "kdca.csv"))
df_snu = pd.read_csv(os.path.join(data, "snu.csv"))

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


# 각 데이터셋을 FAISS 벡터DB로 저장(최초 1회)
def prepare_faiss():
    os.makedirs("vector_db", exist_ok=True)
    # -- 1200_v1
    texts_1200 = df_1200["label"].astype(str) + "\n" + df_1200["text"].astype(str)
    chunks_1200 = chunk_by_words(texts_1200.tolist(), chunk_size=60)
    db_1200 = FAISS.from_texts(chunks_1200, embedding=embedding_model)
    db_1200.save_local("vector_db/faiss_db_1200_v1")

    # -- amc
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

    # -- daily
    daily_texts = df_daily["증상"].astype(str) + "\n" + df_daily["일상말"].astype(str)
    chunks_daily = chunk_by_words(daily_texts.tolist(), chunk_size=60)
    db_daily = FAISS.from_texts(chunks_daily, embedding=embedding_model)
    db_daily.save_local("vector_db/faiss_db_daily_dataset")

    # -- final_v7
    final_texts = df_final["label"].astype(str) + "\n" + df_final["text"].astype(str)
    chunks_final = chunk_by_words(final_texts.tolist(), chunk_size=60)
    db_final = FAISS.from_texts(chunks_final, embedding=embedding_model)
    db_final.save_local("vector_db/faiss_db_final_v7")

    # -- kdca
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

    # -- snu
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


# 최초 1회만 실행
if not os.path.exists("vector_db/faiss_db_1200_v1/index.faiss"):
    prepare_faiss()

# 벡터DB 경로 리스트 및 로드
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

    # 각 DB에서 top-3 유사 문서 검색
    top_k_per_db = 3
    all_docs = []
    for db in db_list:
        all_docs.extend(db.similarity_search_with_score(rag_query, k=top_k_per_db))
    all_docs_sorted = sorted(all_docs, key=lambda x: x[1])
    seen = set()
    context_candidates = []
    for doc, score in all_docs_sorted:
        text = (doc.page_content or "").strip()
        if len(text) < 30:
            continue
        if text in seen:
            continue
        context_candidates.append(text)
        seen.add(text)
        if len(context_candidates) == 3:
            break

    # context(증상 정보) 최종 선택
    retrieved_context = (
        "\n---\n".join(context_candidates)[:1100] if context_candidates else ""
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

    # 프롬프트 생성 및 메시지 구성
    dynamic_prompt = make_prompt(user_question, retrieved_context)
    messages = []
    messages.append({"role": "system", "content": dynamic_prompt})
    for m in memory.chat_memory.messages:
        role = "user" if m.type == "human" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": user_question})

    print("\n[EXAONE-32B 응답]\n", end="", flush=True)  # 응답 시작 줄바꿈

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

    # 항상 존댓말 후처리
    answer_only = make_respectful(answer_only)
    # 2~5번 콤마 등으로 한 줄에 여러 항목이 있는 경우 자동 줄바꿈
    answer_only = format_multiline_sections(answer_only)

    # 대화내역에 assistant 답변 추가
    session.append("assistant", answer_only)
