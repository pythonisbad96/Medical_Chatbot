from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import re

# ✅ LoRA 학습된 모델 로드
base_model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
adapter_path = "./lora_exaone_adapter_v1"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✅ 병명 후처리 함수
def extract_disease(text, prompt):
    response = text.replace(prompt, "").strip()
    first_line = response.split("\n")[0]
    disease = re.split(r"[ \(\.,]", first_line)[0]
    return disease

# ✅ 1. 자동 테스트 (10개 증상 예측)
test_symptoms = [
    "기침이 계속 나고 가래가 많아요",
    "열이 나고 근육통이 심해요",
    "두통이 있고 빛에 예민해요",
    "속이 메스껍고 구토가 있어요",
    "목이 따갑고 음식을 삼키기 힘들어요",
    "밤마다 숨이 차고 기침이 심해요",
    "손발이 저리고 혈당이 불안정해요",
    "소변을 자주 보고 통증이 있어요",
    "눈이 가렵고 재채기가 자주 나요",
    "복통과 설사가 며칠째 지속돼요"
]

print("🔍 [자동 테스트: 증상 10개]")
for i, symptom in enumerate(test_symptoms, 1):
    prompt = f"증상: {symptom}\n질병:"
    result = llm(prompt, max_new_tokens=5, do_sample=False)[0]["generated_text"]
    disease = extract_disease(result, prompt)
    print(f"{i}. 증상: {symptom}")
    print(f"   🩺 예측 병명: {disease}\n")

# ✅ 2. 사용자 입력 루프
print("🧑‍⚕️ 사용자 증상을 입력하세요. (종료하려면 Ctrl + C)\n")
try:
    while True:
        user_input = input("👤 증상 입력: ")
        prompt = f"증상: {user_input}\n질병:"
        result = llm(prompt, max_new_tokens=5, do_sample=False)[0]["generated_text"]
        disease = extract_disease(result, prompt)
        print(f"   🩺 예측 병명: {disease}\n")
except KeyboardInterrupt:
    print("\n👋 종료합니다. 건강하세요!")
