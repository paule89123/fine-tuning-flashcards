import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1", load_in_4bit=True, torch_dtype=torch.float16, device_map="auto")

model = prepare_model_for_kbit_training(model)
tokenizer.pad_token = "!"

CUTOFF_LEN = 128
LORA_R = 1
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.1

config = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=[
                    "w1", "w2", "w3"], lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)

# Load the tiny dataset
dataset = load_dataset("json", data_files="tiny_flashcards_dataset.json")
train_data = dataset["train"]

def generate_prompt(sample):
    sys_msg = f"Create flashcards based on the following prompt: {sample['prompt']}"
    flashcards = "\n".join([f"{{\"front\": \"{card['front']}\", \"back\": \"{card['back']}\"}}" for card in sample["output"]])
    p = f"<s> [INST]{sys_msg}[/INST]\n{flashcards}</s>"
    return p

tokenize = lambda prompt: tokenizer(prompt + tokenizer.eos_token, truncation=True, max_length=CUTOFF_LEN, padding="max_length")
train_data = train_data.map(lambda x: tokenize(generate_prompt(x)), remove_columns=["prompt", "output"])

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=1e-4,
        logging_steps=2,
        optim="adamw_torch",
        save_strategy="epoch",
        output_dir="mixtral-flashcards-lora"
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()