# train an LLM with MMA dataset only on the autoformalization task
from transformers import TrainerCallback, TrainingArguments
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from peft import LoraConfig
import os
from datetime import datetime


MODEL_NAME = "codellama/CodeLlama-7b-hf"

# load MMA dataset
data_files = {
    'train': 'data/MMA dataset/isabelle_train.json',
    'validation': 'data/MMA dataset/isabelle_val.json'
}
dataset = load_dataset('json', data_files=data_files)


wandb_project = f"mma-codellama"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

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
        text = example['input'][i] + '\n' + example['output'][i]
        output_texts.append(text)
    return output_texts


args = TrainingArguments(
    output_dir=f"peft/MMA/{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
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
    run_name=f"MMA-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
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