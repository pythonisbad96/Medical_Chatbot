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

# 환경변수에서 API 설정 (없으면 기본값)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://guest-api.sktax.chat/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sktax-XyeKFrq67ZjS4EpsDlrHHXV8it")
MODEL_NAME = os.getenv("MODEL_NAME", "ax4")

# 데이터 폴더/경로 지정
data = "./dataset"

# 데이터 불러오기
df_1200 = pd.read_csv(os.path.join(data, "1200_v1.csv"))
df_amc = pd.read_csv(os.path.join(data, "amc.csv"))
df_daily = pd.read_csv(os.path.join(data, "daily_dataset.csv"))
df_final = pd.read_csv(os.path.join(data, "final_v7.csv"))
df_kdca = pd.read_csv(os.path.join(data, "kdca.csv"))
df_snu = pd.read_csv(os.path.join(data, "snu.csv"))

# 임베딩 모델 준비 (최초 1회)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")


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


def prepare_faiss():
    os.makedirs("vector_db", exist_ok=True)
    texts_1200 = df_1200["label"].astype(str) + "\n" + df_1200["text"].astype(str)
    chunks_1200 = chunk_by_words(texts_1200.tolist(), chunk_size=60)
    db_1200 = FAISS.from_texts(chunks_1200, embedding=embedding_model)
    db_1200.save_local("vector_db/faiss_db_1200_v1")

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

    daily_texts = df_daily["증상"].astype(str) + "\n" + df_daily["일상말"].astype(str)
    chunks_daily = chunk_by_words(daily_texts.tolist(), chunk_size=60)
    db_daily = FAISS.from_texts(chunks_daily, embedding=embedding_model)
    db_daily.save_local("vector_db/faiss_db_daily_dataset")

    final_texts = df_final["label"].astype(str) + "\n" + df_final["text"].astype(str)
    chunks_final = chunk_by_words(final_texts.tolist(), chunk_size=60)
    db_final = FAISS.from_texts(chunks_final, embedding=embedding_model)
    db_final.save_local("vector_db/faiss_db_final_v7")

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


if not os.path.exists("vector_db/faiss_db_1200_v1/index.faiss"):
    prepare_faiss()

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


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def make_prompt(user_question, retrieved_context):
    return f"""
당신은 의료 상담 챗봇입니다.

모든 답변은 반드시 존댓말(-입니다, -합니다, -해야 합니다, -하시기 바랍니다 등)로만 작성하세요.
내부 생각이나 불확실한 표현 없이, 아래 [증상 정보]를 참고하여 요구된 항목을 **최종 답변만** 작성하세요.

질문이 의료 관련일 경우 [증상 정보]를 참고하여 아래 1~5번을 작성하고,
음식, 여행, 상식 등 **의료와 무관할 경우**에는 [증상 정보]를 무시하고 간결하게 답하세요.

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


def build_prompt(dynamic_prompt):
    return ChatPromptTemplate.from_messages(
        [
            ("system", dynamic_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )


client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

session = ChatSession()
print(f"새로운 세션이 시작되었습니다. session_id: {session.session_id}\n")


def make_respectful(text):
    # 간단한 존댓말 변환 예시. 필요시 패턴 추가
    text = re.sub(r"([^.가-힣\w])?(\w+)[ ]*다([^\w가-힣])", r"\1\2니다\3", text)
    text = re.sub(r"임\.?", "입니다.", text)
    text = re.sub(r"임니다", "입니다", text)
    return text


def format_multiline_sections(text):
    # 2~5번 항목을 여러 줄로 정리(콤마, ·, ㆍ 등 기준)
    section_pat = re.compile(r"([2-5]\..*?:)(.*?)(?=\n[3-5]\.|$)", re.DOTALL)

    def _split_lines(match):
        title = match.group(1)
        content = match.group(2)
        # 콤마, ·, ㆍ 등으로 분리
        # 이미 여러 줄이면 그대로 두고, 한 줄 내에 여러 항목이 있으면 개행 처리
        lines = []
        for item in re.split(r",|·|ㆍ", content):
            item = item.strip(" \n-")
            if not item:
                continue
            if not item.startswith("-"):
                lines.append(f"   - {item}")
            else:
                lines.append(f"   {item}")
        return title + "\n" + "\n".join(lines) + "\n"

    return section_pat.sub(_split_lines, text)


while True:
    user_question = input("질문을 입력하세요(종료: exit): ")
    if user_question.lower() in ["exit", "quit", "종료"]:
        print("프로그램을 종료합니다.")
        session.save()
        break

    prev_utts = [h["message"] for h in session.history if h["role"] == "user"]
    rag_query = (
        " ".join(prev_utts[-2:] + [user_question]) if prev_utts else user_question
    )

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

    retrieved_context = (
        "\n---\n".join(context_candidates)[:1100] if context_candidates else ""
    )

    session.append("user", user_question)
    session.history = session.history[-4:]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for h in session.history:
        if h["role"] == "user":
            memory.chat_memory.add_user_message(h["message"])
        else:
            memory.chat_memory.add_ai_message(h["message"])

    dynamic_prompt = make_prompt(user_question, retrieved_context)

    messages = []
    messages.append({"role": "system", "content": dynamic_prompt})
    for m in memory.chat_memory.messages:
        role = "user" if m.type == "human" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": user_question})

    print("\n[AX-4.0 응답]\n", end="", flush=True)  # 줄바꿈 한 번 추가!
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

    matches = re.findall(
        r"1\..*?\n2\..*?\n3\..*?\n4\..*?\n5\..*(?=\n|$)", answer, flags=re.DOTALL
    )
    answer_only = matches[-1].strip() if matches else answer

    # 1번 항목 줄바꿈 포맷(여러 줄)
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
            merged_bullets = "\n   ".join(bullet_lines)
            section1_fixed = lines[0] + "\n   " + merged_bullets
            answer_only = section1_fixed + section2plus

    # 항상 존댓말로
    answer_only = make_respectful(answer_only)
    # 2~5번 콤마, 점 등으로 한 줄에 나오는 항목은 한 줄씩 줄바꿈 처리
    answer_only = format_multiline_sections(answer_only)

    session.append("assistant", answer_only)
