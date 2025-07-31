import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling

# ✅ Step 1. 데이터 불러오기
df1 = pd.read_csv("final_v7.csv")
df2 = pd.read_csv("1200_v1.csv")
df = pd.concat([df1, df2], ignore_index=True)
df = df[['text', 'label']].dropna()
df['prompt'] = df['text'].apply(lambda x: f"증상: {x}\n질병:")
df['completion'] = df['label']
dataset = Dataset.from_pandas(df[['prompt', 'completion']])

# ✅ Step 2. 모델/토크나이저 로딩 (LoRA 지원 구조)
model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# ✅ Step 3. LoRA 구성
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # EXAONE 구조에 따라 달라질 수 있음
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ✅ Step 4. 토큰화
def tokenize(example):
    full_text = example['prompt'] + " " + example['completion']
    return tokenizer(full_text, truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize)

# ✅ Step 5. 학습 설정
training_args = TrainingArguments(
    output_dir="./lora_exaone_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ Step 6. Trainer로 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ✅ Step 7. LoRA adapter 저장
model.save_pretrained("./lora_exaone_adapter_v1")
tokenizer.save_pretrained("./lora_exaone_adapter_v1")
