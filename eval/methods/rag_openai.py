# generate proofs for the minif2f test set using OpenAI API with RAG
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain

from tqdm import tqdm

import os
import json
import re


GPT35 = True
api_key = "sk-<your-api-key>"

llm = ChatOpenAI(
    # openai_api_base="<base-url>",
    api_key=api_key
)
embeddings = OpenAIEmbeddings(
    # openai_api_base="<base-url>",
    api_key=api_key
)

# load vector dataset
if os.path.exists("minif2f_valid_faiss_index"):
    vector = FAISS.load_local("minif2f_valid_faiss_index", embeddings, allow_dangerous_deserialization=True)
# construct using OpenAI embeddings
else:
    # extract the provided proofs from minif2f valid set
    TEST_FILE = 'minif2f/lean/src/valid.lean'
    SAVE_DIR = 'data/minif2f_valid'
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    dataset = open(TEST_FILE)
    pattern = r'theorem((?:.|\n)*?)end'
    problems = [x.group() for x in re.finditer(pattern, dataset.read())]
    for prob in problems:
        if 'sorry' in prob:
            continue
        theorem_name = prob.split(' ')[1].strip()
        with open(f'{SAVE_DIR}/{theorem_name}.lean', 'w') as f:
            f.write(prob)

    loader = DirectoryLoader('data/minif2f_valid', glob="*.lean")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector  = None
    with tqdm(total=len(docs), desc="Ingesting documents") as pbar:
        for d in docs:
            if vector:
                vector.add_documents([d])
            else:
                vector = FAISS.from_documents([d], embeddings)
            pbar.update(1)  

    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local("minif2f_valid_faiss_index")

prompt = ChatPromptTemplate.from_template("""Answer and proof the following question with code in the Lean 3 formal system, given some related examples from the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

TEST_FILE = 'minif2f/lean/src/test.lean'
USE_LORA = False
SAVE_DIR = "rag/"
save_path = SAVE_DIR + 'gpt35_rag.json'

test_dataset = open(TEST_FILE)
pattern = r'theorem((?:.|\n)*?):='
theorems = [x.group() for x in re.finditer(pattern, test_dataset.read())]

def obj_dict(obj):
    return obj.__dict__

START = 0
results = []
if os.path.exists(save_path):
    results = json.load(open(save_path))
    START = len(results)
for i in tqdm(range(START, len(theorems))):
    theorem = theorems[i]
    response = retrieval_chain.invoke({"input": theorem})
    results.append(json.dumps(response, default=obj_dict))
    json.dump(results, open(save_path, 'w'))
