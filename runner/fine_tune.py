# runner/fine_tune.py
"""
Unsloth Fine-Tuning Script for Adaptive Crisis AI
Optimized for Google Colab (T4 / A100 GPUs)

Usage on Colab:
1. Upload `logs/finetuning_dataset.jsonl`
2. pip install unsloth "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
3. Run this script.
"""

import json
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# --- CONFIGURATION ---
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DATASET_PATH = "logs/finetuning_dataset.jsonl"
OUTPUT_DIR = "outputs/llama3_crisis_lora"

def format_chat_template(example, tokenizer):
    """Formats the messages array into the Llama-3 chat template"""
    messages = example["messages"]
    example["text"] = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return example

def main():
    print("--- Starting Phase 3: Fine-Tuning with Unsloth ---")
    
    # 1. Load Model & Tokenizer
    print(f"Loading base model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None, # Auto-detect
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    print("Injecting LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Rank
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32, # Increased for stronger learning signal
        lora_dropout = 0, 
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Apply Chat Template to Tokenizer
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant", "system": "system"}
    )

    # 4. Load & Prepare Dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    print("Formatting dataset to Llama-3 ChatML...")
    dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), batched=False)

    # 5. Training Setup
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False, # Can make training faster for short sequences
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 3, # 3-5 recommended
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = OUTPUT_DIR,
        ),
    )

    # 6. Start Training
    print("--- Starting Training ---")
    trainer_stats = trainer.train()
    print("--- Training Complete ---")

    # 7. Save Model
    print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR) # Local saving
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n✅ Phase 3 Fine-Tuning Complete!")
    print("To use the model, load it with PEFT or unsloth's FastLanguageModel.from_pretrained()")

if __name__ == "__main__":
    main()
