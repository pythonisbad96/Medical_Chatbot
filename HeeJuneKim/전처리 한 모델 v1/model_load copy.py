from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# 모델 및 토크나이저 로딩
base = AutoModelForCausalLM.from_pretrained(
    "LGAI-EXAONE/EXAONE-4.0-1.2B",
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base, "./lora_exaone_adapter")
tokenizer = AutoTokenizer.from_pretrained("./lora_exaone_adapter", trust_remote_code=True)

# pipeline 생성
llm = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 테스트할 증상 리스트
symptoms = [
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

# 모델 예측
for i, symptom in enumerate(symptoms, 1):
    prompt = f"증상: {symptom}\n질병:"
    result = llm(prompt, max_new_tokens=5, do_sample=False)
    output = result[0]['generated_text'].replace(prompt, "").strip()
    print(f"{i}. 증상: {symptom}")
    print(f"   🔍 예측된 병명: {output}\n")
