# train an LLM with MUSTARDSauce dataset with various experiment settings
from transformers import TrainerCallback, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, concatenate_datasets, Dataset
import torch
from trl import SFTTrainer
from peft import LoraConfig
import os
from datetime import datetime
import json


MODEL_NAME = "codellama/CodeLlama-7b-hf"
# MODEL_NAME = "mistralai/Mistral-7B-v0.1"
CODE_FORMAT = False # if format the code with ```lean\n...\n``` or not
INFORMAL = False # if informal proof is included
ONLY_STATEMENTS = False # if only statements are included, corresponding to autoformalization
MMA = False # if MMA dataset is included
RANDOM = False # if using a random dataset of the same size with also incorrect proofs

# prompt if informal proof is included
prompt_informal = "\n\nFirst answer this question with an informal proof in natural language:\n\n"
# prompts if only statements are included
prompt_auto1 = "Statement in natural language:\n"
prompt_auto2 = "\nTranslate the statement in natural language to Lean:\n"
# prompt for a formal proof in lean
prompt = "\n\nAnswer this question with a formal proof in Lean:\n\n"

# load pre-processed MUSTARDSauce dataset
data_path = 'data/MUSTARDSauce/filtered_the.json' if not RANDOM else 'data/MUSTARDSauce/filtered_rand.json'
dataset = load_dataset('json', data_files=data_path)['train']
print('Mustard dataset length:', len(dataset))

if ONLY_STATEMENTS:
    # exclude wp for word problems that also present in MUSTARDSauce
    exclude_idx = []
    for i in range(len(dataset)):
        if dataset[i]['type'] != 'tp':
            exclude_idx.append(i)
    dataset = dataset.select(
        (
            i for i in range(len(dataset)) 
            if i not in set(exclude_idx)
        )
    )
    print('Mustard dataset length after excluding wp:', len(dataset))
    if MMA:
        file = json.load(open('data/MMA dataset/lean_test.json'))
        prefix = "Statement in natural language:\n"
        suffix = "\nTranslate the statement in natural language to Lean:"
        more_data = {'input': [], 'formal_statement': []}
        for data in file:
            assert data['input'].startswith(prefix)
            assert data['input'].endswith(suffix)
            # extracts the informal and formal statements
            more_data['input'] += [data['input'][len(prefix):-len(suffix)]]
            more_data['formal_statement'] += [data['output']]
        dataset = concatenate_datasets([dataset, Dataset.from_dict(more_data)])
        print('MMA dataset length:', len(more_data['input']))
        dataset = dataset.shuffle(seed=42)
    
dataset = dataset.train_test_split(test_size=0.1)

wandb_project = f"mustard-codellama"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

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

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = example['input'][i]
        if ONLY_STATEMENTS:
            text = prompt_auto1 + example['input'][i] + prompt_auto2 + '```lean\n' + example['formal_statement'][i] + '\n```'
        else:
            if INFORMAL:
                text += prompt_informal + example['intermediate'][i]
            if CODE_FORMAT:
                text += prompt + '```lean\n' + example['output'][i] +  '\n```'
            else:
                text += prompt + example['output'][i]
        output_texts.append(text)
    return output_texts


args = TrainingArguments(
    output_dir=f"peft/mustard/{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    logging_strategy="steps",
    logging_steps=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    save_total_limit=1,
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    learning_rate=5e-4,
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    dataloader_num_workers=4,
    report_to="wandb",
    run_name=f"mustard-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_accumulation_steps=8,
    local_rank=0,
    deepspeed="config/zero3.json",

)

train_dataset = Dataset.from_dict(dataset['train'])
eval_dataset = Dataset.from_dict(dataset['validation'])

trainer = SFTTrainer(
    MODEL_NAME,
    args=args,
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset, 
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
    peft_config=peft_config,
    callbacks=[PeftSavingCallback()],
)
trainer.train()