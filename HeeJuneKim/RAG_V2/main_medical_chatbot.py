import os
import json
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from openai import OpenAI
from rapidfuzz import fuzz

# ✅ 임베딩 모델 로드
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

# ✅ JSON 파일 폴더 경로
json_folder = "./json_diseases_final"

# ✅ 텍스트 데이터 추출 함수
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

# ✅ 벡터 저장소 구축
db_path = "vector_db/faiss_db_json"
os.makedirs("vector_db", exist_ok=True)
texts = []

for file in os.listdir(json_folder):
    if not file.endswith(".json"):
        continue
    with open(os.path.join(json_folder, file), encoding="utf-8") as f:
        data = json.load(f)

    disease_name = data.get("병명", "")
    정의 = extract_text(data.get("정의", {}))
    원인 = extract_text(data.get("원인", {}))
    증상 = extract_text(data.get("증상", {}))
    진단 = extract_text(data.get("진단", {}))
    치료 = extract_text(data.get("치료", {}))

    full_text = f"""
[병명] {disease_name}
[정의] {정의}
[원인] {원인}
[증상] {증상}
[진단] {진단}
[치료] {치료}
""".strip()

    if full_text:
        texts.append(full_text)

if not os.path.exists(db_path + "/index.faiss"):
    db = FAISS.from_texts(texts, embedding=embedding_model)
    db.save_local(db_path)

# ✅ 벡터 DB 로드
db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# ✅ SKT A.X-4.0 LLM
client = OpenAI(
    base_url="https://guest-api.sktax.chat/v1",
    api_key="sktax-XyeKFrq67ZjS4EpsDlrHHXV8it"
)

# ✅ 사용자 입력 루프
while True:
    user_input = input("\n🩺 증상을 입력하세요 (종료: exit): ")
    if user_input.lower() in ["exit", "종료", "quit"]:
        print("챗봇을 종료합니다.")
        break

    # 🔍 유사 질병 검색
    docs = db.similarity_search(user_input, k=10)
    for i, doc in enumerate(docs[:3]):
        score = fuzz.token_sort_ratio(user_input, doc.page_content)
        print(f"{i+1}. 점수: {score} → {doc.page_content[:100]}...\n")

    retrieved_context = "\n---\n".join([doc.page_content for doc in docs])[:600]

    # ✅ 개선된 프롬프트
    prompt = f"""
당신은 의료 상담 챗봇입니다.

사용자 질문이 **건강/증상/의학 관련이면**, 아래 [증상 정보]를 참고하여 1~5번 항목을 작성하세요.  
**하지만** 질문이 음식, 여행, 생활 상식 등 **의료와 무관한 경우**에는 [증상 정보]를 무시하고 자유롭게 답변하세요.  

항상 존댓말(-입니다, -합니다)로 답변하며, 내부 생각 없이 **최종 답변만 출력**하세요.

---

🧠 질문: {user_input}

---

[증상 정보]
{retrieved_context}

---

📝 출력 형식:
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

    # 🧠 모델 호출
    response = client.chat.completions.create(
        model="ax4",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip()

    # ✅ 출력 (RAG 형식일 경우만 1~5 추출)
    if "1." in answer and "2." in answer:
        match = re.search(r"1\..*?5\..*", answer, flags=re.DOTALL)
        answer_only = match.group().strip() if match else answer
        answer_only = re.sub(r"(습니다|합니다)\1+", r"\1", answer_only)
    else:
        answer_only = answer.strip()

    print("\n🧾 [SKT A.X-4.0 응답 결과]")
    print(answer_only)
