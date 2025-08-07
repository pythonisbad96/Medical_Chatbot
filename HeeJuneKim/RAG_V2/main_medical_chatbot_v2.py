import os
import json
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
from rapidfuzz import fuzz

# ① SBERT 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# ② JSON 데이터 폴더 및 FAISS 저장 경로
json_folder = "./json_diseases_final"
db_path = "vector_db/faiss_db_json"
os.makedirs("vector_db", exist_ok=True)

# ③ dict 필드 추출 함수
def extract_text(field_dict):
    if not isinstance(field_dict, dict):
        return ""
    texts = []
    for v in field_dict.values():
        if isinstance(v, list):
            texts.append("\n".join(map(str, v)))
        elif isinstance(v, str):
            texts.append(v)
        else:
            texts.append(str(v))
    return "\n".join(texts)

# ④ 인덱스가 없을 경우 FAISS 생성
if not os.path.exists(os.path.join(db_path, "index.faiss")):
    texts = []
    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(json_folder, filename), encoding="utf-8") as f:
            data = json.load(f)
        disease = data.get("병명", "")
        정의 = extract_text(data.get("정의", {}))
        원인 = extract_text(data.get("원인", {}))
        증상 = extract_text(data.get("증상", {}))
        진단 = extract_text(data.get("진단", {}))
        치료 = extract_text(data.get("치료", {}))
        full_text = f"[병명] {disease}\n[정의] {정의}\n[원인] {원인}\n[증상] {증상}\n[진단] {진단}\n[치료] {치료}"
        if full_text:
            texts.append(full_text)
    db = FAISS.from_texts(texts, embedding=embedding_model)
    db.save_local(db_path)

# ⑤ 벡터 DB 로드
db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# ⑥ LLM 클라이언트 초기화
client = OpenAI(
    base_url="https://guest-api.sktax.chat/v1",
    api_key="sktax-XyeKFrq67ZjS4EpsDlrHHXV8it"
)

# ⑦ 시스템 프롬프트 설정 (역할 + 응답 포맷)
system_content = """
당신은 의료 상담 챗봇입니다.

사용자 질문이 건강/증상/의학 관련이면, 아래 [증상 정보]를 참고하여 1~5번 항목을 작성하세요.
반드시 존댓말(-입니다, -합니다)로 답변하며, 내부 생각은 출력하지 않습니다.

비의료 질문(음식, 여행 등)이면 [증상 정보]를 무시하고 자유롭게 답변하세요.

📝 출력 형식:
1. 예상되는 병명 (2~3가지): (첫 번째 병명은 간단한 설명도 포함)
2. 주요 원인:
3. 추천 진료과 (2~3과):
4. 예방 및 관리 방법:
5. 생활 시 주의사항:

(비의료 질문일 경우)
답변:
""".strip()

# ⑧ 멀티턴 메시지 초기화
messages = [
    {"role": "system", "content": system_content}
]

# ⑨ 사용자 입력 루프
retrieved_context_added = False  # 증상 정보는 최초 1회만 추가

while True:
    user_input = input("\n🩺 증상을 입력하세요 (종료: exit): ")
    if user_input.lower() in ["exit", "종료", "quit"]:
        print("챗봇을 종료합니다.")
        break

    # 🔍 유사도 검색 (RAG 컨텍스트 생성) → 최초 1회만
    if not retrieved_context_added:
        docs = db.similarity_search(user_input, k=10)
        print("\n🔍 유사 증상 상위 3개:")
        for i, doc in enumerate(docs[:3]):
            score = fuzz.token_sort_ratio(user_input, doc.page_content)
            snippet = doc.page_content[:100].replace("\n", " ")
            print(f"{i+1}. 점수: {score} → {snippet}...\n")

        retrieved_context = "\n---\n".join([doc.page_content for doc in docs])[:1000]
        messages.append({
            "role": "system",
            "content": f"[증상 정보]\n{retrieved_context}"
        })
        retrieved_context_added = True

    # 🙋‍♀️ 사용자 질문만 추가 (단순 문장)
    messages.append({"role": "user", "content": user_input})

    # 🤖 모델 호출
    response = client.chat.completions.create(
        model="ax4",
        messages=messages
    )
    answer = response.choices[0].message.content.strip()

    # ✅ 1~5 항목만 추출
    if "1." in answer and "2." in answer:
        match = re.search(r"1\..*?5\..*", answer, flags=re.DOTALL)
        answer_only = match.group().strip() if match else answer
        answer_only = re.sub(r"(입니다|합니다)\1+", r"\1", answer_only)
    else:
        answer_only = answer

    # 💬 모델 응답도 누적
    messages.append({"role": "assistant", "content": answer_only})

    # 📤 출력
    print("\n🧾 [SKT A.X-4.0 응답 결과]")
    print(answer_only)
