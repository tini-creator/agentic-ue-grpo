import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Configuration
MODEL_DIR = "../models/sft-ue-merged"  # Load the merged base model
DEVICE = "cpu"

print(f"Loading model from {MODEL_DIR} to {DEVICE}...")

# 2. Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map={"": DEVICE},
    dtype=torch.float32
)

# 3. Define test scenarios
test_contexts = [
    "Battery is at 10%. User is reading a basic text article.",
    "Battery is at 95%. User launched a heavy 3D mobile game.",
    "Battery is at 45%. Device is downloading a large system update in the background."
]

print("\n--- Starting Inference ---\n")

# 4. The Generation Loop
for context in test_contexts:
    # A. Format the prompt EXACTLY as it was in SFT
    prompt = f"UE Context: {context}\nDevice Config:\n"

    # B. Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # C. Generate the output
    with torch.no_grad():  # we aren't training right now
        outputs = model.generate(
            **inputs,
            max_new_tokens=60,  # Don't let it ramble, JSON is short
            temperature=0.1,  # Low temperature = highly deterministic/logical output
            do_sample=True,  # Required if using temperature
            pad_token_id=tokenizer.eos_token_id
        )

    # D. Decode the generated tokens back into text
    # We slice [inputs.input_ids.shape[1]:] to only print the NEW text, not the prompt
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print(f"PROMPT:  {context}")
    print(f"OUTPUT:  {generated_text.strip()}")
    print("-" * 50)