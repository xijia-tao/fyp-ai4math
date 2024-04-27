# generate model responses for the minif2f test set under self refine method
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig
import os
import json
import re
from tqdm import tqdm

# prompts for self refine method
prompt = "Answer this question with a formal proof in Lean:\n\n```lean\n"
feedback_prompt = "\nThere is an error in the code above because of lack of understanding of the question. What is the error? To find the error, go through semantically complete blocks of the code, and check if everything looks good.\n"
refine_prompt = "\nOkay! Here is the rewrite:\n"

MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TEST_FILE = 'minif2f/lean/src/test.lean'
BATCH_SIZE = 4
USE_LORA = False # if using LoRA
peft = 'peft/mustard/<checkpoint>' # replace with a specific checkpoint location if using LoRA

SAVE_DIR = 'sr-minif2f-results/'
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
    exp = 'mistral-7b' if 'mistral' in MODEL_NAME else 'codellama-7b'
    save_path = SAVE_DIR + exp + '.json'

# load the first round of results
first_round = json.load(open(save_path.replace('sr-minif2f', 'minif2f')))
results = []
for i in tqdm(range(0, len(theorems), BATCH_SIZE)):

     # otherwise generate from scratch
    # xs = [prompt + theorem for theorem in theorems[i:i+BATCH_SIZE]]
    # inputs = tokenizer(xs, padding=True, return_tensors="pt").to(device=model.device)
    # with torch.no_grad():
    #     outputs = model.generate(**inputs, max_new_tokens=500) #, temperature=0.1)
    # ys = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ys = first_round[i:i+BATCH_SIZE]

    xs = [y + feedback_prompt for y in ys]
    inputs = tokenizer(xs, padding=True, return_tensors="pt").to(device=model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)
    fs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    xs = [f + refine_prompt for f in fs]
    inputs = tokenizer(xs, padding=True, return_tensors="pt").to(device=model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)
    rs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results += rs

    json.dump(results, open(save_path, 'w'))
    