import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 1. Configuration
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
DATA_FILE = "../data/train_data.jsonl"
OUTPUT_DIR = "../models/sft-ue-baseline"

print("Loading dataset and formatting prompts...")

# 2. Load the dataset
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# 3. Define the exact format the LLM needs to learn
def formatting_prompts_func(example):
    # Combine prompt and response into a single string under a new "text" column
    example["text"] = f"UE Context: {example['prompt']}\nDevice Config:\n{example['response']}<|endoftext|>"
    return example

# Apply the formatting function to all rows
dataset = dataset.map(formatting_prompts_func, remove_columns=["prompt", "response"])
print(f"Loading Base Model ({MODEL_ID}) to CPU...")

# 4. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Critical for batching

# 5. Load Model (Strictly configured for CPU)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": "cpu"},       # Force CPU execution
    dtype=torch.float32     # Standard precision for CPU stability
)

print("Applying LoRA (Low-Rank Adaptation)...")

# 6. Configure LoRA
# We freeze the 135M parameters and only train tiny adapter matrices
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # Target the attention mechanisms
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

#model = get_peft_model(model, lora_config)
#model.print_trainable_parameters()

print("Configuring CPU-friendly Training Arguments...")

# 7. CPU-Optimized Training Arguments using the new SFTConfig
training_args = SFTConfig(
    use_liger_kernel=False,
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # Process 1 example at a time in memory
    gradient_accumulation_steps=4,      # Wait for 4 examples before doing the math update
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    optim="adamw_torch",
    report_to="none",
    max_length=128,                     # Limits text chunks to save CPU memory
    use_cpu=True,         # <--- Explicitly tell it to use CPU
    fp16=False,           # <--- Disable GPU-specific 16-bit float
    bf16=False,            # <--- Disable GPU-specific bfloat16
    dataset_text_field="text"
)

# 8. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,                 # Pass the SFTConfig here
    peft_config=lora_config
)

# 9. Execute Training
print("Starting Supervised Fine-Tuning on CPU. This will take a few minutes...")
trainer.train()

# 10. Save the final model
print(f"Training complete! Saving LoRA adapter to {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")