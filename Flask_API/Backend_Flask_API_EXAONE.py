# -*- coding: utf-8 -*-
"""
medical_server.py
- 기존 medical_main.py의 기능을 Flask RESTful API로 변환
- API 엔드포인트: /ask_symptoms
"""

# ------------------------------------------------------------
# 0) 표준/서드파티 모듈 임포트
# ------------------------------------------------------------
import os, json, re
from typing import List, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 임포트

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

# GPU 설정: torch가 있으면 CUDA 사용, 없으면 CPU 사용
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# ------------------------------------------------------------
# 1) 경로/DB 기본 설정
# ------------------------------------------------------------
JSON_FOLDER = os.getenv("JSON_FOLDER", "./json_diseases_final_ver").strip()
_default_dbdir = f"vector_unified_{os.path.basename(JSON_FOLDER) or 'db'}"
DB_DIR = os.getenv("DB_DIR", _default_dbdir).strip()

UNIFIED_DB_PATH = f"{DB_DIR}/faiss_unified_disease_db"
os.makedirs(DB_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2) 실행 옵션 및 기준값(Threshold) 설정
# ------------------------------------------------------------
FORCE_REBUILD = os.getenv("FORCE_REBUILD", "0") == "1"
K_DISEASE = int(os.getenv("K_DISEASE", "10"))
MAX_DISEASES = int(os.getenv("MAX_DISEASES", "5"))
CTX_CHARS = int(os.getenv("CTX_CHARS", "4000"))

LOW_CONF_THRESHOLD = 0.5
HIGH_CONF_THRESHOLD = 0.74
SCORE_DIFF_THRESHOLD = 0.03

# ------------------------------------------------------------
# 3) 임베딩 모델 준비
# ------------------------------------------------------------
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask").strip()
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
)

# ------------------------------------------------------------
# 4) FriendliAI LLM 클라이언트 준비 (OpenAI 호환)
# ------------------------------------------------------------
FRIENDLI_BASE_URL = "https://api.friendli.ai/serverless/v1"
FRIENDLI_MODEL = os.getenv("FRIENDLI_MODEL", "LGAI-EXAONE/EXAONE-4.0.1-32B").strip()

FRIENDLI_TOKEN = os.getenv("FRIENDLI_TOKEN")
if not FRIENDLI_TOKEN:
    raise RuntimeError(
        "환경변수 FRIENDLI_TOKEN 이(가) 비었습니다. .env 또는 OS 환경변수에 설정하세요."
    )

llm_client = OpenAI(
    api_key=FRIENDLI_TOKEN,
    base_url=FRIENDLI_BASE_URL,
)


def chat_with_friendli(messages, **gen_opts) -> str:
    completion = llm_client.chat.completions.create(
        model=FRIENDLI_MODEL,
        messages=messages,
        temperature=gen_opts.get(
            "temperature", 0.3
        ),  # 숫자를 늘리면 늘릴수록 창의적인 답변이 나옴.
        max_tokens=gen_opts.get("max_tokens", 1024),
    )
    return completion.choices[0].message.content.strip()


# ------------------------------------------------------------
# 5) LLM 보조 유틸리티 함수들
# ------------------------------------------------------------
def extract_any(section_val) -> str:
    if section_val is None:
        return ""
    if isinstance(section_val, dict):
        texts = []
        if "supplement" in section_val:
            supp = section_val.get("supplement")
            if supp and isinstance(supp, list):
                texts.append("\n".join(str(x) for x in supp if x and str(x) != "None"))
        for k, v in section_val.items():
            if k == "supplement":
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
    return getattr(doc, "metadata", {}).get("병명", "알 수 없는 질병")


def extract_numbered_block(answer: str) -> str:
    items = re.findall(r"(?ms)^\s*\d\.\s.*?(?=^\s*\d\.|\Z)", answer)
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


def extract_diagnosis_parts(answer: str) -> dict:
    """
    LLM의 답변을 항목별로 분리하여 딕셔너리로 반환.
    `extract_numbered_block`을 확장한 버전.
    """
    out = {
        "predictedDiagnosis": "",
        "diagnosisDefinition": "",
        "recommendedDepartment": "",
        "preventionManagement": "",
        "additionalInfo": "",
        "medicine": "",  # ✅ 추가: 상비약 추천을 위한 새로운 필드
        "rawResponse": answer,
    }

    blocks = re.split(r"\n\s*(?=\d+\.\s)", answer.strip())

    for block in blocks:
        lines = block.strip().split("\n", 1)
        title = lines[0].strip().replace(":", "").replace("：", "")
        body = lines[1].strip() if len(lines) > 1 else ""

        if "예상되는 병명" in title or "예측 병명" in title:
            out["predictedDiagnosis"] = body
        elif "주요 원인" in title:
            out["diagnosisDefinition"] = body
        elif "추천 진료과" in title:
            out["recommendedDepartment"] = body
        elif "예방 및 관리 방법" in title:
            out["preventionManagement"] = body
        elif "생활 시 주의사항" in title:
            out["additionalInfo"] = body
        elif "상비약 추천" in title:
            out["medicine"] = body  # ✅ 수정: additionalInfo 대신 medicine에 저장

    return out


# ------------------------------------------------------------
# 6) FAISS 헬퍼 및 단일 인덱스 구축 함수
# ------------------------------------------------------------
def faiss_from_texts(texts, embedding_model, metadatas=None):
    try:
        return FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    except TypeError:
        return FAISS.from_texts(texts, embeddings=embedding_model, metadatas=metadatas)


def faiss_load_local(path, embedding_model):
    try:
        return FAISS.load_local(
            path, embedding=embedding_model, allow_dangerous_deserialization=True
        )
    except TypeError:
        return FAISS.load_local(
            path, embeddings=embedding_model, allow_dangerous_deserialization=True
        )


def _index_file(path: str) -> str:
    return os.path.join(path, "index.faiss")


def _latest_json_mtime(folder: str) -> float:
    times = []
    if not os.path.isdir(folder):
        return 0.0
    for f in os.listdir(folder):
        if f.lower().endswith(".json"):
            times.append(os.path.getmtime(os.path.join(folder, f)))
    return max(times) if times else 0.0


def _needs_rebuild(index_path: str, source_folder: str) -> bool:
    if FORCE_REBUILD:
        return True
    idx = _index_file(index_path)
    if not os.path.exists(idx):
        return True
    return os.path.getmtime(idx) < _latest_json_mtime(source_folder)


def build_or_load_unified_disease_db():
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
            supplement_text = ""
            if isinstance(symptom_data, dict) and "supplement" in symptom_data:
                supplement_text = extract_any(symptom_data.get("supplement"))

            weighted_symptom_part = (
                f"[증상] {symptom_text}\n" + f"[증상.supplement] {supplement_text}\n"
            ) * 3

            other_info_parts = [f"[병명] {disease_name}"]
            for key, value in data.items():
                if key not in ["병명", "증상"]:
                    content = extract_any(value)
                    if content:
                        other_info_parts.append(f"[{key}] {content}")
            other_info_part = "\n".join(other_info_parts)
            weighted_document_text = (weighted_symptom_part + other_info_part).strip()
            texts_for_embedding.append(weighted_document_text)

            clean_symptom_part = f"[증상] {symptom_text}"
            clean_document_text = (clean_symptom_part + "\n" + other_info_part).strip()
            metas.append(
                {
                    "병명": disease_name,
                    "파일": filename,
                    "clean_text": clean_document_text,
                }
            )

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
def search_unified_db_with_scores(
    db, user_query: str, k: int
) -> List[Tuple[any, float]]:
    q_fmt = f"query: {user_query}"
    if not db:
        return []
    return db.similarity_search_with_score(q_fmt, k)


# ------------------------------------------------------------
# 8) 프롬프트(SYSTEM) 및 Flask 앱
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

app = Flask(__name__)
CORS(app)  # 모든 경로에 대해 CORS 허용

try:
    disease_db = build_or_load_unified_disease_db()
    print("✅ 통합 인덱스 준비 완료")
except Exception as e:
    print(f"❌ 통합 인덱스 로드/빌드 실패: {e}")
    disease_db = None


@app.route("/ask_symptoms", methods=["POST"])
def ask_symptoms():
    if disease_db is None:
        return jsonify({"error": "백엔드 시스템이 준비되지 않았습니다."}), 503

    data = request.get_json()
    user_input = data.get("symptom")
    additional_symptoms = data.get("additional_symptoms", "")

    # ⭐ 수정: 환자 기본 정보를 파싱
    patient_info = data.get("patient", {})
    age = patient_info.get("age")
    gender = patient_info.get("gender")
    conditions = patient_info.get("conditions")

    if not user_input:
        return jsonify({"error": "symptom 필드가 요청 본문에 필요합니다."}), 400

    # ⭐ 수정: 환자 정보를 포함하여 검색 쿼리 및 프롬프트에 활용할 문자열 생성
    patient_prefix = ""
    if age or gender or conditions:
        info_parts = []
        if age:
            info_parts.append(f"{age}세")
        if gender == "m":
            info_parts.append("남자")
        if gender == "f":
            info_parts.append("여자")
        if conditions and conditions != "없음":
            info_parts.append(f"기저질환: {conditions}")
        if info_parts:
            patient_prefix = f"환자 정보: {' '.join(info_parts)}. "

    combined_input = (
        f"{patient_prefix}{user_input}\n추가 정보: {additional_symptoms}"
        if additional_symptoms
        else f"{patient_prefix}{user_input}"
    )

    # ⭐ 추가: 디버그를 위해 입력받은 증상과 환자 정보 출력
    print("\n" + "=" * 50)
    print("⭐ 새 요청 처리 시작")
    print(f"  환자 정보: 나이={age}, 성별={gender}, 기저질환='{conditions}'")
    print(
        f"  입력된 증상: '{user_input}'"
        + (f" + 추가 증상: '{additional_symptoms}'" if additional_symptoms else "")
    )
    print(f"  최종 검색 쿼리: '{combined_input}'")
    print("-" * 50)

    # 1) 검색 수행 (combined_input 사용)
    docs_with_scores = search_unified_db_with_scores(
        disease_db, combined_input, k=K_DISEASE
    )

    # 2) 검색 실패 시 일반 응답 라우팅
    if not docs_with_scores:
        print("[Info] 관련 질병 정보를 찾을 수 없습니다. 일반적인 답변을 시도합니다.")
        general_messages = [
            {
                "role": "system",
                "content": "당신은 사용자에게 친절하게 답변하는 AI 어시스턴트입니다.",
            },
            {"role": "user", "content": combined_input},
        ]
        try:
            answer = chat_with_friendli(general_messages)
            # ✅ 수정: 원본 답변을 바로 반환
            return jsonify({"answer": answer})
        except Exception as e:
            return jsonify({"error": f"API 호출 실패: {e}"}), 500

    # 3) 거리 -> 간이 유사도 변환
    docs_with_sim_scores = [(doc, 1 / (1 + score)) for doc, score in docs_with_scores]
    unique_docs = docs_with_sim_scores

    # 4) 상위 1개 유사도 점수로 라우팅 판단
    top1_doc, top1_score = unique_docs[0]

    # (A) 비의료/잡담 라우팅
    if top1_score < LOW_CONF_THRESHOLD:
        print(f"[판단] 비의료 질문 (유사도: {top1_score:.2f} < {LOW_CONF_THRESHOLD})")
        general_messages = [
            {
                "role": "system",
                "content": "당신은 사용자에게 친절하게 답변하는 AI 어시스턴트입니다.",
            },
            {"role": "user", "content": combined_input},
        ]
        try:
            answer = chat_with_friendli(general_messages)
            # ✅ 수정: 원본 답변을 바로 반환
            print(f"[디버그] 분리된 답변: {answer}")
            return jsonify({"answer": answer})
        except Exception as e:
            return jsonify({"error": f"API 호출 실패: {e}"}), 500

    # (B) 확신도 판단
    is_confident = top1_score >= HIGH_CONF_THRESHOLD and (
        len(unique_docs) < 3
        or (unique_docs[0][1] - unique_docs[2][1]) >= SCORE_DIFF_THRESHOLD
    )

    if not is_confident and not additional_symptoms:
        print(f"[판단] 확신도 낮음 (유사도: {top1_score:.2f}). 추가 증상 요청.")
        return jsonify(
            {
                "status": "needs_more_info",
                "message": "증상을 조금 더 구체적으로 알려주시겠어요? 추가적인 증상이 있다면 함께 입력해주세요.",
            }
        )

    # (C) 확신도가 높거나, 추가 증상이 이미 있다면 최종 답변 생성
    print(f"[판단] 확신도 높음 또는 추가 정보로 재검색. (유사도: {top1_score:.2f})")
    final_docs = [doc for doc, score in unique_docs[:MAX_DISEASES]]

    # 5) LLM 생성
    if final_docs:
        final_context = "\n---\n".join(
            [doc.metadata.get("clean_text", doc.page_content) for doc in final_docs]
        )

        # ⭐ 수정: 환자 정보를 LLM 프롬프트에 추가
        llm_user_query = f"{patient_prefix}사용자 질문: {combined_input}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"[질병 참고]\n{final_context[:CTX_CHARS]}"},
            {"role": "user", "content": llm_user_query},
        ]
        try:
            answer = chat_with_friendli(messages)
            structured_answer = extract_diagnosis_parts(answer)
            print(f"[디버그] 분리된 답변: {structured_answer}")
            return jsonify({"answer": structured_answer})
        except Exception as e:
            return jsonify({"error": f"최종 답변 생성 실패: {e}"}), 500

    return jsonify({"error": "알 수 없는 오류가 발생했습니다."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
