import os
import json
import re
import ast
import uuid
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.memory import ConversationBufferMemory
from openai import OpenAI

# 0. 환경변수 / 모델 설정
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://guest-api.sktax.chat/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sktax-XyeKFrq67ZjS4EpsDlrHHXV8it")
MODEL_NAME = os.getenv("MODEL_NAME", "ax4")

# 데이터 경로
json_dir = "./top100_json"                 # 100개 질병 JSON
mfds_path = "../dataset/medicine/mfds.csv" # 의약품 CSV


# 1. 전처리 유틸
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_value(v):
    """
    리스트/딕트/문자열을 안전하게 문자열로 통합.
    문자열 안에 리스트 형태("['A','B']")가 들어오면 ast.literal_eval 시도.
    """
    if v is None:
        return ""
    if isinstance(v, (list, tuple, set)):
        return " / ".join([normalize_text(x) for x in v if x])
    if isinstance(v, dict):
        return " / ".join([normalize_text(x) for x in v.values() if x])
    s = str(v).strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple, set)):
                return " / ".join([normalize_text(x) for x in parsed if x])
        except Exception:
            pass
    return normalize_text(s)

# 2. 데이터 로딩
def load_json_dataset(json_dir):
    """
    질병 JSON(100개) → 텍스트 리스트
    사용 키: 병명, 정의, 원인, 진단, 치료, 진료과 + 증상(하위 필드 포함)
    """
    all_texts = []
    if not os.path.isdir(json_dir):
        return all_texts

    keys_flat = ["병명", "정의", "원인", "진단", "치료", "진료과"]

    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue
        with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        parts = []
        for k in keys_flat:
            parts.append(normalize_value(data.get(k)))

        symptoms = data.get("증상")
        if symptoms:
            if isinstance(symptoms, dict):
                for sv in symptoms.values():
                    parts.append(normalize_value(sv))
            else:
                parts.append(normalize_value(symptoms))

        merged = "\n".join([p for p in parts if p]).strip()
        if merged:
            all_texts.append(merged)
    return all_texts

def load_mfds_dataset(csv_path):
    """
    의약품 CSV → 한 행(제품)당 하나의 문서 텍스트
    사용 컬럼: 제품명, 성분목록, 성분영문목록, 효능효과, 용법용량, 상세링크
    """
    texts = []
    if not os.path.isfile(csv_path):
        return texts

    df = pd.read_csv(csv_path, encoding="utf-8")
    for _, row in df.iterrows():
        name     = normalize_value(row.get("제품명", ""))
        ingred   = normalize_value(row.get("성분목록", ""))
        ingred_e = normalize_value(row.get("성분영문목록", ""))
        effect   = normalize_value(row.get("효능효과", ""))
        dosage   = normalize_value(row.get("용법용량", ""))
        link     = normalize_value(row.get("상세링크", ""))

        doc = []
        if name:     doc.append(f"[제품명]\n{name}")
        if ingred:   doc.append(f"[성분]\n{ingred}")
        if ingred_e: doc.append(f"[성분(영문)]\n{ingred_e}")
        if effect:   doc.append(f"[효능효과]\n{effect}")
        if dosage:   doc.append(f"[용법용량]\n{dosage}")
        if link:     doc.append(f"[상세링크]\n{link}")

        merged = "\n".join(doc).strip()
        if merged:
            texts.append(merged)
    return texts

# 3. 임베딩 / 청크 / 인덱싱 (단일 DB)
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts")

def chunk_by_words(texts, chunk_size=60):
    chunks = []
    for text in texts:
        if not text:
            continue
        words = str(text).split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    return chunks

def build_and_save_faiss(texts, save_path, chunk_size=60):
    chunks = chunk_by_words(texts, chunk_size=chunk_size)
    if not chunks:
        return
    db = FAISS.from_texts(chunks, embedding=embedding_model)
    os.makedirs(os.path.dirname(save_path), exist_ok=True
    )
    db.save_local(save_path)

def prepare_faiss_single():
    """
    질병 JSON + 의약품 CSV → 하나의 단일 인덱스: vector_db/faiss_db_medical_all
    """
    index_path = "vector_db/faiss_db_medical_all/index.faiss"
    if os.path.exists(index_path):
        return
    texts_json = load_json_dataset(json_dir)
    texts_mfds = load_mfds_dataset(mfds_path)
    all_texts = (texts_json or []) + (texts_mfds or [])
    build_and_save_faiss(all_texts, "vector_db/faiss_db_medical_all", chunk_size=60)

# 최초 1회 인덱스 생성
prepare_faiss_single()


# 4. 벡터 DB 로드 (하나만)
db_path = "vector_db/faiss_db_medical_all"
db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)

# 5. 세션/프롬프트/후처리
class ChatSession:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.history = []

    def save(self, path="session_logs/"):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{self.session_id}.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def append(self, user, message):
        self.history.append({"role": user, "message": message})

# LangChain 메모리(틀 유지용)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def make_prompt(user_question, ranked_contexts_text):
    """
    ranked_contexts_text: '#1 ...\n---\n#2 ...' 처럼 유사도 순서로 정렬된 컨텍스트
    """
    return f"""
당신은 의료 상담 챗봇입니다.

모든 답변은 반드시 존댓말(-입니다, -합니다, -해야 합니다 등)로만 작성하세요.
아래 [증상 정보]는 유사도 순위대로 정렬(#1이 가장 유사)되어 있습니다.
반드시 유사도 순위대로(1→N) 근거를 반영하세요.

질문이 의료 관련이면 [증상 정보] 참고 1~5번, 무관하면 간결하게만 답변.
---
질문: {user_question}
---
[증상 정보(유사도 순위)]
{ranked_contexts_text}
---
출력 예시(유사도 순위 반영):
1. 예상되는 병명 (각 병명마다 한 줄 설명 포함, 반드시 존댓말, 유사도 순위대로 번호를 붙이세요):
   - (1위) 첫 번째 병명: (설명)
   - (2위) 두 번째 병명: (설명)
   - (3위) 세 번째 병명: (설명)
   - (선택) (4위) 네 번째 병명: (설명)
   - (선택) (5위) 다섯 번째 병명: (설명)
2. 주요 원인:
3. 추천 진료과 (2~3과):
4. 예방 및 관리 방법:
5. 생활 시 주의사항:

(비의료 질문일 경우)
답변:
""".strip()

def make_respectful(text):
    # 간단한 존댓말 보정
    text = re.sub(r"([^.가-힣\w])?(\w+)[ ]*다([^\w가-힣])", r"\1\2니다\3", text)
    text = re.sub(r"임\.?", "입니다.", text)
    text = re.sub(r"임니다", "입니다", text)
    return text

def format_multiline_sections(text):
    # 2~5번 항목 나열 자동 줄바꿈
    section_pat = re.compile(r"([2-5]\..*?:)(.*?)(?=\n[3-5]\.|$)", re.DOTALL)
    def _split_lines(match):
        title = match.group(1)
        content = match.group(2)
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

# 6. LLM 클라이언트
client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)


# 7. 대화 루프 (멀티턴 + 단일 DB RAG + 동적 3/5개 컨텍스트)
# 동적 선택 파라미터
SEARCH_K = 15          # 1차 검색 개수(여유 있게)
GAP_THRESHOLD = 0.08   # 3위와 4위 간 '차이'가 이 값 이상이면 3개만 사용 (값은 경험적으로 조정)

session = ChatSession()
print(f"새로운 세션이 시작되었습니다. session_id: {session.session_id}\n")

while True:
    user_question = input("질문을 입력하세요(종료: exit): ")
    if user_question.lower() in ["exit", "quit", "종료"]:
        print("프로그램을 종료합니다.")
        session.save()
        break

    # 최근 2턴의 사용자 질문까지 합쳐 RAG 쿼리 생성
    prev_utts = [h["message"] for h in session.history if h["role"] == "user"]
    rag_query = (" ".join(prev_utts[-2:] + [user_question]) if prev_utts else user_question)

    # 단일 DB에서 top-N 검색
    docs_with_scores = db.similarity_search_with_score(rag_query, k=SEARCH_K)

    # 점수 기준 정렬: FAISS는 일반적으로 "낮을수록 유사(거리)" → 오름차순 정렬
    docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])

    # 텍스트 기준 중복 제거(유지: 유사도 순서)
    uniq_texts = []
    uniq_scores = []
    seen = set()
    for doc, score in docs_with_scores:
        text = (doc.page_content or "").strip()
        if len(text) < 30:
            continue
        if text in seen:
            continue
        uniq_texts.append(text)
        uniq_scores.append(score)
        seen.add(text)
        if len(uniq_texts) >= 10:
            break  # 충분

    # 동적 3/5개 결정 로직
    use_n = 3
    if len(uniq_scores) >= 5:
        # 3위와 4위 사이의 '격차' 확인
        s3 = float(uniq_scores[2])
        s4 = float(uniq_scores[3])
        # 거리(낮을수록 좋음) 가정: 차이가 작으면 5개, 크면 3개
        gap = abs(s4 - s3)
        # 상대적 차이(스케일 보정)
        rel_gap = gap / max(1e-9, abs(s3))
        use_n = 5 if rel_gap < GAP_THRESHOLD else 3
    else:
        use_n = min(3, len(uniq_texts))

    chosen_texts = uniq_texts[:use_n]
    chosen_scores = uniq_scores[:use_n]

    # 유사도 순위 주석(#1, #2 ...) 달아서 프롬프트에 넣기
    ranked_blocks = []
    for i, (t, s) in enumerate(zip(chosen_texts, chosen_scores), start=1):
        ranked_blocks.append(f"# {i}\n(검색점수: {s:.6f})\n{t}")
    ranked_contexts_text = "\n---\n".join(ranked_blocks)

    # 대화 히스토리 최신 2쌍 유지
    session.append("user", user_question)
    session.history = session.history[-4:]

    # LangChain 메모리 갱신
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for h in session.history:
        if h["role"] == "user":
            memory.chat_memory.add_user_message(h["message"])
        else:
            memory.chat_memory.add_ai_message(h["message"])

    # 프롬프트 구성 (유사도 순위 반영 지시 포함)
    dynamic_prompt = make_prompt(user_question, ranked_contexts_text)
    messages = [{"role": "system", "content": dynamic_prompt}]
    for m in memory.chat_memory.messages:
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        messages.append({"role": role, "content": m.content})
    messages.append({"role": "user", "content": user_question})

    print("\n[AX-4.0 응답]\n", end="", flush=True)

    # 스트리밍 호출
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
    matches = re.findall(r"1\..*?\n2\..*?\n3\..*?\n4\..*?\n5\..*(?=\n|$)", answer, flags=re.DOTALL)
    answer_only = matches[-1].strip() if matches else answer

    # 1번(병명들) 줄바꿈 정리
    pattern = r"(1\..*?)(\n2\..*)"
    m = re.search(pattern, answer_only, flags=re.DOTALL)
    if m:
        section1 = m.group(1)
        section2plus = m.group(2)
        lines = section1.split("\n")
        bullet_lines = [line.rstrip() for line in lines if line.strip().startswith("- ")]
        if bullet_lines:
            merged_bullets = "\n   ".join(bullet_lines)
            section1_fixed = lines[0] + "\n   " + merged_bullets
            answer_only = section1_fixed + section2plus

    # 존댓말/가독성 보정
    answer_only = make_respectful(answer_only)
    answer_only = format_multiline_sections(answer_only)

    # 세션에 assistant 답변 저장
    session.append("assistant", answer_only)
