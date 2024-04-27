# generate proofs for the minif2f test set using OpenAI's GPT-3.5 or GPT-4 API
from openai import OpenAI
import json
import re
from tqdm import tqdm
import os


GPT35 = True # False for GPT-4
TEST_FILE = 'minif2f/lean/src/test.lean'

# extract minif2f test theorems
test_dataset = open(TEST_FILE)
pattern = r'theorem((?:.|\n)*?):='
theorems = [x.group() for x in re.finditer(pattern, test_dataset.read())]


prompt = "Prove the following theorem in the Lean 3 formal system. Give me only the code without explanation.\n" 

SAVE_DIR = 'openai/'
save_to = 'gpt35_minif2f.json' if GPT35 else 'gpt4_minif2f.json'
save_to = SAVE_DIR + save_to
model_name = "gpt-3.5-turbo" if GPT35 else "gpt-4-turbo-preview"
api_key = "sk-<your-api-key>"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
    # base_url="<base-url>"
)

results = []

def gpt_api(messages: list):
    completion = client.chat.completions.create(model=model_name, messages=messages)
    print(completion)
    print(completion.choices[0].message.content)
    results.append(completion.choices[0].message.content)
    json.dump(results, open(save_to, 'w'), indent=4)


for i in tqdm(len(theorems)):
    inp = prompt + theorems[i]
    messages = [{'role': 'user','content': inp},]
    gpt_api(messages)
