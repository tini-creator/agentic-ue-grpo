"""
RL fine-tuning for the UE resource manager — migrated to TRL 1.x
-----------------------------------------------------------------
TRL 1.x removed AutoModelForCausalLMWithValueHead, PPOConfig, and PPOTrainer
from the top-level API (they live in trl.experimental.ppo now and are slated
for removal in 0.29.0).

The idiomatic TRL 1.x replacement is GRPOTrainer, which:
  - Drops the value/critic head entirely (no AutoModelForCausalLMWithValueHead)
  - Accepts a plain callable reward_funcs instead of a manual step() loop
  - Is more memory-efficient — better for CPU-only training
  - Handles generation, KL penalty, and weight updates internally

pip install trl>=1.3.0 transformers peft datasets torch
"""

import os
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Reward function
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.reward_model import calculate_ue_reward

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "sft-ue-merged"))
DATA_FILE   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data",   "train_data.jsonl"))
OUTPUT_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "grpo-ue-agent"))
METRICS_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "metrics", "grpo_metrics.json"))

print("Loading data and model (CPU mode) …")

# ---------------------------------------------------------------------------
# 2. Tokenizer
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# 3. Dataset
#    GRPOTrainer expects a "prompt" column (string).
#    We build the same prefixed prompt as before so the reward function
#    still receives the full context it needs.
# ---------------------------------------------------------------------------
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def build_prompt(example):
    example["prompt"] = (
        f"UE Context: {example['prompt']}\n"
        f"Required keys (USE UNDERSCORES, NEVER HYPHENS!): "
        f"power_mode, drx_cycle, offload_compute, max_bandwidth_mbps\n"
        f"Device Config:\n{{"
    )
    return example

dataset = raw_dataset.map(build_prompt)

# ---------------------------------------------------------------------------
# 4. Reward function
#    GRPOTrainer calls reward_funcs with (prompts, completions) as lists.
#    Return a list of floats — one per sample in the batch.
# ---------------------------------------------------------------------------
def ue_reward_fn(prompts, completions, **kwargs):
    """Wrap the scalar reward model for GRPOTrainer's list-based interface.
    verbose=True prints a terminal block per sample showing context, raw
    output, any JSON repair applied, and the final reward (green = positive,
    red = negative).  Set verbose=False to silence during eval/inference.
    """
    return [
        calculate_ue_reward(prompt, completion, verbose=True)
        for prompt, completion in zip(prompts, completions)
    ]

# ---------------------------------------------------------------------------
# 5. LoRA config (same targets as before)
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# ---------------------------------------------------------------------------
# 6. GRPOConfig
#    Maps the old PPOConfig knobs onto their GRPO equivalents:
#      batch_size=4                  → per_device_train_batch_size=4
#      mini_batch_size=1             → (handled internally by num_mini_batches)
#      gradient_accumulation_steps=4 → gradient_accumulation_steps=4
#      learning_rate=5e-5            → learning_rate=5e-5
#      ppo_epochs=4                  → num_train_epochs=4
#      init_kl_coef=0.15             → beta=0.15
#
#    Generation kwargs are passed via the generation_config dict;
#    bad_words_ids is no longer used — the model is better-constrained
#    by the SFT step and the reward signal, and bad_words can suppress
#    valid sub-tokens (see code-review note).
# ---------------------------------------------------------------------------
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,

    # Optimiser
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=4,

    # GRPO-specific
    num_generations=4,          # completions sampled per prompt (replaces mini_batch_size logic)
    beta=0.15,                  # KL penalty coefficient (was init_kl_coef)
    loss_type="grpo",           # "grpo" (default) or "bnpo" / "dr_grpo"

    # Generation
    max_completion_length=120,
    temperature=0.2,
    top_k=5,
    top_p=0.85,

    # Logging / saving
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",           # disable W&B

    # CPU-friendly
    bf16=False,
    fp16=False,
)

# ---------------------------------------------------------------------------
# 7. Trainer
#    - model: path string or loaded model — GRPOTrainer loads it internally
#    - peft_config: applied automatically (no manual get_peft_model needed)
#    - reward_funcs: callable or list of callables
# ---------------------------------------------------------------------------
trainer = GRPOTrainer(
    model=MODEL_DIR,
    args=grpo_config,
    train_dataset=dataset,
    reward_funcs=ue_reward_fn,
    peft_config=lora_config,
    processing_class=tokenizer,
)

# ---------------------------------------------------------------------------
# 8. Train
# ---------------------------------------------------------------------------
print("\n--- Starting GRPO Reinforcement Learning ---\n")
train_result = trainer.train()

# ---------------------------------------------------------------------------
# 9. Save model
# ---------------------------------------------------------------------------
print(f"\nSaving RL agent to {OUTPUT_DIR} …")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# 10. Persist training metrics (all GRPO keys, not just reward)
# ---------------------------------------------------------------------------
log_history = trainer.state.log_history

GRPO_METRIC_KEYS = [
    "reward", "reward_std", "kl", "loss", "grad_norm", "frac_reward_zero_std"
]
training_history = [
    {
        "step": entry.get("step", i),
        **{k: entry[k] for k in GRPO_METRIC_KEYS if k in entry},
    }
    for i, entry in enumerate(log_history)
    if any(k in entry for k in GRPO_METRIC_KEYS)
]

os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(training_history, f, indent=4)
print(f"Training metrics saved to {METRICS_FILE}")

# ---------------------------------------------------------------------------
# 11. Post-training KPI evaluation  (CVR + OOD)
#     Runs the same test sets as evaluate.py but inline, so the numbers
#     are written into ppo_metrics.json alongside the reward curve for
#     plot_rewards.py to pick up.
#
#     Import is deferred to here so the file can be run without evaluate.py
#     present (e.g. during a quick smoke-test).
# ---------------------------------------------------------------------------
print("\n--- Post-training KPI evaluation ---")

try:
    import importlib.util, types

    # ── load evaluate.py from the scripts directory ───────────────────────
    eval_path = os.path.normpath(os.path.join(SCRIPT_DIR, "evaluate.py"))
    spec = importlib.util.spec_from_file_location("evaluate", eval_path)
    ev   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    # ── build test sets ───────────────────────────────────────────────────
    id_set     = ev.build_id_test_set(n=150)
    rule_a_set = ev.build_rule_a_set(n=80)
    ood_set    = ev.build_ood_test_set()

    # ── load the just-saved GRPO model for inference ──────────────────────
    from transformers import AutoModelForCausalLM
    eval_model = AutoModelForCausalLM.from_pretrained(
        OUTPUT_DIR, torch_dtype=torch.float32, device_map="cpu"
    )
    eval_model.eval()

    def _get_config(prompt_text):
        return ev.generate_config(eval_model, tokenizer, prompt_text)

    # ── CVR ───────────────────────────────────────────────────────────────
    cvr_violations = sum(
        1 for s in rule_a_set
        if ev.is_rule_a_violation(_get_config(s["prompt"]), s["battery"])
    )
    cvr = cvr_violations / len(rule_a_set)

    # ── Config Accuracy (ID + OOD) ────────────────────────────────────────
    id_hits  = sum(1 for s in id_set  if ev.exact_match(_get_config(s["prompt"]), s["label"]))
    ood_hits = sum(1 for s in ood_set if ev.exact_match(_get_config(s["prompt"]), s["label"]))
    id_acc   = id_hits  / len(id_set)
    ood_acc  = ood_hits / len(ood_set)
    ood_gap  = id_acc - ood_acc

    kpi_summary = {
        "cvr":          round(cvr,     4),
        "id_accuracy":  round(id_acc,  4),
        "ood_accuracy": round(ood_acc, 4),
        "ood_gap":      round(ood_gap, 4),
        "n_rule_a":     len(rule_a_set),
        "n_id":         len(id_set),
        "n_ood":        len(ood_set),
    }

    print(f"  CVR              : {cvr:.1%}  (target → 0%)")
    print(f"  ID  Accuracy     : {id_acc:.1%}")
    print(f"  OOD Accuracy     : {ood_acc:.1%}")
    print(f"  OOD Gap          : {ood_gap:.1%}  (smaller = better generalisation)")

    # ── append KPI snapshot to metrics file ──────────────────────────────
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)

    existing.append({"event": "post_training_kpi", **kpi_summary})

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)

    print(f"KPI snapshot appended to {METRICS_FILE}")

except FileNotFoundError:
    print("  [SKIP] evaluate.py not found alongside run_ppo.py — skipping KPI eval.")
except Exception as exc:
    print(f"  [WARN] KPI evaluation failed: {exc}")
    print("         Run scripts/evaluate.py manually after training.")

print("\nComplete!")
