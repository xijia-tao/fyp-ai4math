# generate model responses for the minif2f test set under corex method
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from peft import LoraConfig
import os
import json
import re
from tqdm import tqdm

# prompts for corex method
init_ftp_system_prompt = {
        "Tom" : "You are Tom, an expert renowned for exceptional math skills and knowledge of formal theorem proving in Lean 3 formal system. Your friends often approach you for help with formal theorem proving due to your ability to explain complex concepts in an easy-to-understand manner. Your task here is to draw upon your deep understanding of mathematical concepts to answer the subsequent question. Please explain your proof in Lean 3 step by step, demonstrating your thought process clearly.",
        "Jerry" : "You are Jerry, a postgraduate math student renowned for exceptional math skills and knowledge of formal theorem proving in Lean 3 formal system. Your teachers are always impressed with your ability in rigorously proving a theorem. Your task here is to draw upon your deep understanding of mathematical concepts to answer the subsequent question. Please explain your proof in Lean 3 step by step, demonstrating your thought process clearly.",
}

ftp_debate = {
        "Tom" : """You are engaged in a friendly discussion with your friend Jerry regarding a math theorem to prove in Lean 3. Jerry shared his solution: {}. As you review it, your task is to carefully consider Jerry's insights, incorporate any valuable elements into your own analysis, and refine your proof accordingly. Remember, this is a collaborative endeavor aimed at reaching the best possible proof together. Please articulate your response thoughtfully.""",
        "Jerry" : """You are in the middle of a friendly discussion with your friend Tom about a complex math theorem to prove in Lean 3. Tom has just presented his solution: {}. Your role now is to thoughtfully evaluate Tom's approach, incorporate any beneficial aspects into your own, and adjust your proof to enhance its quality. Keep in mind, this is a collaborative endeavor where both of you are working towards reaching the best proof. Please craft your response carefully."""
}

MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
TEST_FILE = 'minif2f/lean/src/test.lean'
BATCH_SIZE = 4
USE_LORA = False # if using LoRA
peft = 'peft/mustard/<checkpoint>' # replace with a specific checkpoint location if using LoRA

SAVE_DIR = 'ma-minif2f-results/'
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


results = []
for i in tqdm(range(0, len(theorems), BATCH_SIZE)):
    result = {}

    # generate round 1
    tom1 = [init_ftp_system_prompt["Tom"] + '\n' + theorem for theorem in theorems[i:i+BATCH_SIZE]]
    jerry1 = [init_ftp_system_prompt["Jerry"] + '\n' + theorem for theorem in theorems[i:i+BATCH_SIZE]]
    inputs = tokenizer(tom1 + jerry1, padding=True, return_tensors="pt").to(device=model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500) #, temperature=0.1)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result['r1'] = {
        'Tom': output_texts[:BATCH_SIZE],
        'Jerry': output_texts[BATCH_SIZE:]
    }

    # strip prompt from output_texts
    tom_out = ['\n'.join(result['r1']['Tom'][j].split('\n')[1:]).strip() for j in range(BATCH_SIZE)]
    jerry_out = ['\n'.join(result['r1']['Jerry'][j].split('\n')[1:]).strip() for j in range(BATCH_SIZE)]
    
    # generate round 2
    tom2 = [result['r1']['Tom'][j] + '\n' + ftp_debate["Tom"].format(jerry_out[j]) for j in range(BATCH_SIZE)]
    jerry2 = [result['r1']['Jerry'][j] + '\n' + ftp_debate["Jerry"].format(tom_out[j]) for j in range(BATCH_SIZE)]
    inputs = tokenizer(tom2 + jerry2, padding=True, return_tensors="pt").to(device=model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300) #, temperature=0.1)
    output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    result['r2'] = {
        'Tom': output_texts[:BATCH_SIZE],
        'Jerry': output_texts[BATCH_SIZE:]
    }

    results.append(result)
    json.dump(results, open(save_path, 'w'))
    