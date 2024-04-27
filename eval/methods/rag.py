# generate theorems from minif2f test theorems using RAG
from tqdm import tqdm

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig
import os
import json
import re

# reuse the retrieved context when exp with openai models
file = json.load(open('rag/gpt35_rag_clean.json'))
MAX_CONTEXTS = 100 # 2
contexts = []
for data in file:
    context = ['```lean\n' + c['page_content'] + '\n```' for c in data['context']]
    context = context[:MAX_CONTEXTS]
    contexts.append(context)

template = """Answer and proof the following question with code in the Lean 3 formal system, given some related examples within <context> </context>:

<context>
{context}
</context>

Question: ```lean\n{inp}"""

MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TEST_FILE = 'minif2f/lean/src/test.lean'
BATCH_SIZE = 4
USE_LORA = True # if using LoRA
peft = 'peft/mustard/<checkpoint>'

SAVE_DIR = "rag/"
exp = 'mistral-7b' if 'mistral' in MODEL_NAME else 'codellama-7b'

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

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
if USE_LORA:
    model.load_adapter(peft)
    save_path = SAVE_DIR + peft.split('/')[-2] + '.json'
else:
    save_path = SAVE_DIR + exp + '.json'

START = 0
results = []
if os.path.exists(save_path):
    results = json.load(open(save_path))
    START = len(results)
for i in tqdm(range(START, len(theorems), BATCH_SIZE)):
    inps = []
    for j in range(i, min(i+BATCH_SIZE, len(theorems))):
        inp = template.format(context='\n\n'.join(contexts[j]), inp=theorems[j])
        inps.append(inp)

    inputs = tokenizer(inps, padding=True, return_tensors="pt").to(device=model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results += output_texts
    json.dump(results, open(save_path, 'w'))

