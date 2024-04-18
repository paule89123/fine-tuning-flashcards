import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load the fine-tuned model and tokenizer
model_path = "./mixtral-flashcards-lora/checkpoint-6"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
# model = PeftModel.from_pretrained(model, model_path)

# Set the pad token
tokenizer.pad_token = "!"

# Function to generate an article based on a given prompt
def generate_flashcards(prompt):
    sys_msg = "You are an assistant that helps users create flashcards based on the content they provide. Your task is to generate a set of flashcards in JSON format, where each flashcard has a 'front' (question or term) and a 'back' (answer or definition). The number of flashcards generated should match the user's request. Aim to create concise, informative, and well-structured flashcards that effectively capture the key points of the given content. Here is the user's prompt: "
    prompt = f"<s> [INST]{sys_msg}\n{prompt}[/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        num_beams=4,
        temperature=0.8,
        no_repeat_ngram_size=3,
        early_stopping=True,
        do_sample=True
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    flashcards = generated_text.split("[/INST]")[-1].strip()
    return flashcards

# Example usage
prompt = "Create 3 flashcards about the solar system"
generated_flashcards = generate_flashcards(prompt)
print("Prompt:", prompt)
print("Generated flashcards:")
print(generated_flashcards)