# generate model responses for the minif2f test set under standard prompting
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig
import json
import re
import os
from tqdm import tqdm


# prompt for a formal proof in lean
prompt = "\n\nAnswer this question with a formal proof in Lean:\n\n"

MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TEST_FILE = 'minif2f/lean/src/test.lean'
BATCH_SIZE = 4
USE_LORA = False # if using LoRA
peft = 'peft/mustard/<checkpoint>' # replace with a specific checkpoint location if using LoRA

SAVE_DIR = 'minif2f-results/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# extract minif2f test theorems
test_dataset = open(TEST_FILE)
pattern = r'theorem((?:.|\n)*?):='
theorems = [x.group() for x in re.finditer(pattern, test_dataset.read())]


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
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


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config) # load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
if USE_LORA:
    model.load_adapter(peft)
    save_path = SAVE_DIR + peft.split('/')[-2] + '.json'
else:
    save_path = SAVE_DIR + 'mistral-7b.json' if 'mistral' in MODEL_NAME else SAVE_DIR + 'codellama-7b.json'

results = []
for theorem in tqdm(range(0, len(theorems), BATCH_SIZE)):
    batch_theorems = theorems[theorem:theorem+BATCH_SIZE]
    
    texts = [prompt + theorem for theorem in batch_theorems]
    inputs = tokenizer(texts, padding=True, return_tensors="pt").to(device=model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=500) #, temperature=0.1)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    results += output_texts
    json.dump(results, open(save_path, 'w'))
