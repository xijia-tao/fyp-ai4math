# perform inference with local LLMs with command-line interaction
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig


MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
USE_LORA = True # if using LoRA
peft = 'peft/mustard/<checkpoint>' # replace with a specific checkpoint location if using LoRA
prompt = "\nAnswer this question with a formal proof in Lean:\n"
prompt_auto1 = "Statement in natural language:\n"
prompt_auto2 = "\nTranslate the statement in natural language to Lean, and then complete the proof:\n"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto') 
if USE_LORA:
    model.load_adapter(peft)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

while True:
    text = input("Enter a theorem to be proved (the instruction will be added automatically): ")
    text = text + prompt
    inputs = tokenizer(text, return_tensors="pt").to('cuda')

    if USE_LORA:
        model.enable_adapters()
    output = model.generate(**inputs, max_length=500, temperature=0.1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)
    print('-'*50)
