# generate model responses for the minif2f test set under reflexion method
# assume that minif2f-results has already been generated under standard prompting
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig
import os
import json
import re
from tqdm import tqdm
import subprocess


# prompts for the reflexion method
prompt = "Answer this question with a formal proof in Lean:\n\n```lean\n"
eval_prompt = "\nFor the previous proof, the tactic state and/or info from the Lean 3 formal system is:\n"
reflect_prompt = "\nSo the error in the previous proof is"
final_prompt = "\nReflect on the feedback and rewrite the answer in Lean:\n\n```lean\n" 

MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME =   "mistralai/Mistral-7B-v0.1" 
TEST_FILE = 'minif2f/lean/src/test.lean'
BATCH_SIZE = 4
USE_LORA = True # if using LoRA
peft = 'peft/mustard/<checkpoint>' # replace with a specific checkpoint location if using LoRA

SAVE_DIR = 'reflexion-results/'
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

imports = """import minif2f_import

open_locale big_operators
open_locale nat
open_locale real
open_locale rat

"""

pattern = r'```lean((?:.|\n)*?)```'

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config) # load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
if USE_LORA:
    model.load_adapter(peft)
    save_path = SAVE_DIR + peft.split('/')[-2] + '.json'
else:
    save_path = SAVE_DIR + 'mistral-7b.json' if 'mistral' in MODEL_NAME else SAVE_DIR + 'codellama-7b.json'

# load the first round of results
first_round = json.load(open(save_path.replace('reflexion', 'minif2f')))
results = []
for i in tqdm(range(0, len(theorems), BATCH_SIZE)):

    # otherwise generate from scratch
    # xs = [prompt + theorem for theorem in theorems[i:i+BATCH_SIZE]]
    # inputs = tokenizer(xs, padding=True, return_tensors="pt").to(device=model.device)
    # with torch.no_grad():
    #     outputs = model.generate(**inputs, max_new_tokens=500) #, temperature=0.1)
    # ys = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ys = first_round[i:i+BATCH_SIZE]
    
    feedbacks = []
    for i, y in enumerate(ys):
        theorems = [x.group() for x in re.finditer(pattern, y)]
        corrects = []
        if len(theorems) == 0:
            feedback = "No lean proof found. Surround your lean proof with ```lean and ```."
        else:
            # verify the generated proof in the first round with Lean
            the = theorems[0]
            exp = peft.split('/')[-2] if USE_LORA else MODEL_NAME.split('/')[1]
            with open(f'tmp-{exp}.lean', 'w') as f:
                f.write(imports + the)
            try:
                with open(f'tmp-{exp}.txt', 'w') as f:
                    subprocess.run(["lean", f'tmp-{exp}.lean'], stdout=f, timeout=120) 
            except:
                with open(f'tmp-{exp}.txt', 'w') as f:
                    f.write('A timeout with 120-second limit is reached. Check your tactics for their correctness.\n')
            feedback = open(f'tmp-{exp}.txt').read().strip()
            if len(feedback) == 0:
                feedback = "The proof is correct."
                corrects.append(i)
    
            os.remove(f'tmp-{exp}.txt')
            os.remove(f'tmp-{exp}.lean')
        feedbacks.append(feedback)
        

    xs = [y + eval_prompt + feedbacks[i] + final_prompt for i, y in enumerate(ys) if i not in corrects]
    inputs = tokenizer(xs, padding=True, return_tensors="pt").to(device=model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)
    fs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i in range(BATCH_SIZE):
        if i in corrects:
            fs.insert(i, ys[i])

    results += fs

    json.dump(results, open(save_path, 'w'))
    