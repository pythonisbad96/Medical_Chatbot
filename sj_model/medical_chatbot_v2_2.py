# 멀티턴으로 수정 및 일부 코드 수정

import os
import pandas as pd
import glob
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.chat_models import ChatOpenAI   # 사용 안 함(속도 개선용)
# from langchain.chains import LLMChain                    # 사용 안 함(속도 개선용)

from openai import OpenAI  # 팀원 방식: 경량 SDK + 스트리밍

import re
import uuid
import json

# 0. 환경변수에서 API 설정 (없으면 기본값)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://guest-api.sktax.chat/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "sktax-XyeKFrq67ZjS4EpsDlrHHXV8it")
MODEL_NAME      = os.getenv("MODEL_NAME",      "ax4")

# 1. 데이터 폴더/경로 지정
data = "./dataset"

# 2. 데이터 불러오기
df_1200 = pd.read_csv(os.path.join(data, "1200_v1.csv"))
df_amc   = pd.read_csv(os.path.join(data, "amc.csv"))
df_daily = pd.read_csv(os.path.join(data, "daily_dataset.csv"))
df_final = pd.read_csv(os.path.join(data, "final_v7.csv"))
df_kdca  = pd.read_csv(os.path.join(data, "kdca.csv"))
df_snu   = pd.read_csv(os.path.join(data, "snu.csv"))

# 3. 임베딩 모델 준비 (최초 1회)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# 4. 단어 단위 chunking 함수
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

# 5. 각 파일별 chunk + 벡터DB 저장 (최초 1회)
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
        + "\n" + df_amc["정의"].astype(str)
        + "\n" + df_amc["원인"].astype(str)
        + "\n" + df_amc["증상"].astype(str)
        + "\n" + df_amc["진단"].astype(str)
        + "\n" + df_amc["치료"].astype(str)
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
        + "\n" + df_kdca["정의"].astype(str)
        + "\n" + df_kdca["원인"].astype(str)
        + "\n" + df_kdca["증상"].astype(str)
        + "\n" + df_kdca["진단"].astype(str)
        + "\n" + df_kdca["치료"].astype(str)
    )
    chunks_kdca = chunk_by_words(kdca_texts.tolist(), chunk_size=60)
    db_kdca = FAISS.from_texts(chunks_kdca, embedding=embedding_model)
    db_kdca.save_local("vector_db/faiss_db_kdca")
    # snu
    snu_texts = (
        df_snu["병명"].astype(str)
        + "\n" + df_snu["정의"].astype(str)
        + "\n" + df_snu["원인"].astype(str)
        + "\n" + df_snu["증상"].astype(str)
        + "\n" + df_snu["진단/검사"].astype(str)
        + "\n" + df_snu["치료"].astype(str)
    )
    chunks_snu = chunk_by_words(snu_texts.tolist(), chunk_size=60)
    db_snu = FAISS.from_texts(chunks_snu, embedding=embedding_model)
    db_snu.save_local("vector_db/faiss_db_snu")

# 최초 1회만 실행 (이미 저장돼 있으면 생략)
if not os.path.exists("vector_db/faiss_db_1200_v1/index.faiss"):
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

# 7. 대화 세션/상태 관리
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

# 8. LangChain 메모리(틀 유지용) + 프롬프트 생성 함수
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def make_prompt(user_question, retrieved_context):
    return f"""
당신은 의료 상담 챗봇입니다.

사용자 질문이 **건강/증상/의학 관련이면**, 아래 [증상 정보]를 참고하여 1~5번 항목을 작성하세요.  
**하지만** 질문이 음식, 여행, 생활 상식 등 **의료와 무관한 경우**에는 [증상 정보]를 무시하고 자유롭게 답변하세요.  

항상 존댓말(-입니다, -합니다)로 답변하며, 내부 생각 없이 **최종 답변만 출력**하세요.

---
질문: {user_question}
---
[증상 정보]
{retrieved_context}
---
출력 형식:
(의료 질문일 경우)
1. 예상되는 병명 (2~3가지):  
   - 첫 번째 병명은 간단한 설명도 포함해주세요.
2. 주요 원인:
3. 추천 진료과 (2~3과):
4. 예방 및 관리 방법:
5. 생활 시 주의사항:

(비의료 질문일 경우)
답변:
""".strip()

# 프롬프트 템플릿 빌더 — 실제 호출은 OpenAI SDK로
def build_prompt(dynamic_prompt):
    return ChatPromptTemplate.from_messages(
        [
            ("system", dynamic_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

# 9. OpenAI SDK 클라이언트(팀원 방식)
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# --- 실행 루프 ---
session = ChatSession()
print(f"새로운 세션이 시작되었습니다. session_id: {session.session_id}\n")

while True:
    user_question = input("질문을 입력하세요(종료: exit): ")
    if user_question.lower() in ["exit", "quit", "종료"]:
        print("프로그램을 종료합니다.")
        session.save()
        break

    # 1) 멀티턴 기반 RAG 쿼리 (최근 유저 2턴 + 현재 질문)
    prev_utts = [h["message"] for h in session.history if h["role"] == "user"]
    rag_query = " ".join(prev_utts[-2:] + [user_question]) if prev_utts else user_question

    # 2) RAG 컨텍스트 (매 턴 재검색) + 필터링
    top_k_per_db = 3
    all_docs = []
    for db in db_list:
        all_docs.extend(db.similarity_search_with_score(rag_query, k=top_k_per_db))

    all_docs_sorted = sorted(all_docs, key=lambda x: x[1])  # score 낮을수록 유사↑
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

    retrieved_context = "\n---\n".join(context_candidates)[:1100] if context_candidates else ""

    # 3) 세션 히스토리 누적 & 최근 N턴만 유지(틀 유지)
    session.append("user", user_question)
    session.history = session.history[-4:]  # 최근 4개(2쌍)만 유지

    # 4) LangChain 메모리에 최근 히스토리 반영(틀 유지)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for h in session.history:
        if h["role"] == "user":
            memory.chat_memory.add_user_message(h["message"])
        else:
            memory.chat_memory.add_ai_message(h["message"])

    # 5) 시스템 프롬프트(동적) + messages 구성(OpenAI SDK로 직접 호출)
    dynamic_prompt = make_prompt(user_question, retrieved_context)

    messages = []
    # system: 동적 프롬프트(여기에 [증상 정보]도 주입되어 있음)
    messages.append({"role": "system", "content": dynamic_prompt})

    # chat_history: LangChain 메모리에서 꺼내어 OpenAI messages로 변환
    for m in memory.chat_memory.messages:
        role = "user" if m.type == "human" else "assistant"
        messages.append({"role": role, "content": m.content})

    # 현재 user 입력(추가로 원하면 여기에 다시 [증상 정보] 붙일 수도 있으나, system에 이미 포함)
    messages.append({"role": "user", "content": user_question})

    # 6) 스트리밍 호출(팀원 스타일) → 체감 속도 ↑
    print("\n[AX-4.0 응답] ", end="", flush=True)
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
    print()  # 줄바꿈
    answer = "".join(answer_chunks).strip()

    # 7) 정규식 포맷(1~5번 섹션만 추출)
    matches = re.findall(
        r"1\..*?\n2\..*?\n3\..*?\n4\..*?\n5\..*(?=\n|$)", answer, flags=re.DOTALL
    )
    answer_only = matches[-1].strip() if matches else answer

    # 8) assistant 기록 & 출력
    session.append("assistant", answer_only)
    print("\n[정리]\n" + answer_only)
