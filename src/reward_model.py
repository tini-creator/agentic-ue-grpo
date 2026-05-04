import json
import re


# App profiles: maps keyword → (compute_tier, base_bandwidth_mbps)
APP_PROFILES = {
    "podcast":         ("low",       2),
    "system update":   ("low",      20),
    "video":           ("medium",   50),
    "game":            ("high",     15),
    "ar/vr":           ("very_high",100),
}


def parse_context(prompt):
    """Extract battery, app compute tier, and expected bandwidth from the prompt."""
    battery_match = re.search(r'Battery is at (\d+)%', prompt)
    battery = int(battery_match.group(1)) if battery_match else 50

    compute = "low"
    base_bw = 20  # fallback if no app matched

    prompt_lower = prompt.lower()
    for keyword, (tier, bw) in APP_PROFILES.items():
        if keyword in prompt_lower:
            compute = tier
            base_bw = bw
            break

    # Determine expected max_bandwidth given context rules
    if battery <= 20:
        expected_bw = min(base_bw, 5)   # Rule A: throttle on critical battery
    elif "congested" in prompt_lower:
        expected_bw = max(1, base_bw // 2)  # congested network → cap at half
    else:
        expected_bw = base_bw

    return battery, compute, expected_bw


def _print_step(prompt, raw_output, repaired_json, reward, parse_error=None):
    """Pretty-print one training step to the terminal."""
    bat = re.search(r'Battery is at (\d+)%', prompt)
    app = re.search(r'opened a (.+?)\.', prompt)
    net = re.search(r'Network is (\w+)', prompt)
    ctx = "  ".join(filter(None, [
        f"bat={bat.group(1)}%" if bat else "bat=?",
        f"app={app.group(1)}" if app else "app=?",
        f"net={net.group(1)}" if net else "net=?",
    ]))

    bar   = "─" * 64
    color = "\033[92m" if reward >= 0 else "\033[91m"  # green / red
    reset = "\033[0m"

    print(f"\n{bar}")
    print(f"  Context  : {ctx}")
    print(f"  Raw out  : {raw_output[:120].strip()}")
    if repaired_json and repaired_json != raw_output:
        print(f"  Repaired : {repaired_json}")
    if parse_error:
        print(f"  ✗ Parse  : {parse_error}")
    print(f"  Reward   : {color}{reward:+.1f}{reset}")
    print(bar)


def dict_raise_on_duplicates(ordered_pairs):
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError(f"Duplicate key: {k}")
        d[k] = v
    return d


def _repair_json(raw: str) -> str:
    """
    Best-effort repair of the common malformed JSON patterns an LLM produces.
    Operates on a string that has already been confirmed to start with '{'.

    Fixes applied in order:
      1. Python booleans  True/False → true/false  (must run before quote swap
         so we don't accidentally match inside string values)
      2. Python None      None       → null
      3. Single-quoted keys/values   → double-quoted
      4. Unquoted bare-word keys     → double-quoted  (e.g.  {power_mode: …})
      5. Trailing comma before '}'   → removed
      6. Truncate at the first '}'   (flat schema — no nested objects)
      7. Append missing '}'          if the model cut off mid-output
    """
    s = raw

    # 1 & 2 — Python literals (word-boundary match avoids touching string content)
    s = re.sub(r'\bTrue\b',  'true',  s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b',  'null',  s)

    # 3 — single-quoted strings → double-quoted
    #     handles:  'key': 'value'  and  'key': 42
    s = re.sub(r"'([^']*)'", r'"\1"', s)

    # 4 — unquoted bare-word keys  { power_mode: … }  →  { "power_mode": … }
    s = re.sub(r'(?<=[{,])\s*([A-Za-z_][A-Za-z_0-9]*)\s*:', r' "\1":', s)

    # 5 — trailing comma before closing brace
    s = re.sub(r',\s*}', '}', s)

    # 6 — keep only up to and including the first '}'
    if '}' in s:
        s = s.split('}')[0] + '}'
    else:
        # 7 — model cut off before closing brace
        s = s.rstrip() + '\n}'

    return s


def calculate_ue_reward(prompt, generated_text, verbose=True):
    reward = 0.0
    raw = generated_text.strip()

    # ── Step 1: re-attach the '{' that was used as a prefill token ──────────
    if not raw.startswith("{"):
        raw = "{" + raw

    # ── Step 2: repair common LLM JSON mistakes ─────────────────────────────
    clean_json = _repair_json(raw)

    # ── Step 3: parse ────────────────────────────────────────────────────────
    try:
        config = json.loads(clean_json, object_pairs_hook=dict_raise_on_duplicates)
        reward += 2.0
    except ValueError as e:
        if verbose:
            _print_step(prompt, raw, clean_json, reward=-4.0, parse_error=str(e))
        return -4.0

    # 5. Schema check — reward valid keys/values, penalise hallucinations
    required_keys = {"power_mode", "drx_cycle", "offload_compute", "max_bandwidth_mbps"}
    valid_power   = {"max_save", "balanced", "performance"}
    valid_drx     = {"short", "medium", "long"}

    keys_found = set(config.keys())

    for key, value in config.items():
        if key not in required_keys:
            reward -= 2.0
        else:
            reward += 1.0
            if key == "power_mode" and value not in valid_power:
                reward -= 2.0
            elif key == "drx_cycle" and value not in valid_drx:
                reward -= 2.0
            elif key == "offload_compute" and not isinstance(value, bool):
                reward -= 2.0
            elif key == "max_bandwidth_mbps" and not isinstance(value, (int, float)):
                reward -= 2.0

    if keys_found != required_keys:
        return reward  # early exit — incomplete schema, no logic bonus

    reward += 2.0  # passed perfect schema

    # 6. Logic evaluation — now covers all app/battery combinations
    battery, compute, expected_bw = parse_context(prompt)
    power_mode = config["power_mode"]
    drx        = config["drx_cycle"]
    offload    = config["offload_compute"]
    bw         = config["max_bandwidth_mbps"]

    # --- Battery critical (≤ 20%) ---
    if battery <= 20:
        if power_mode == "max_save" and drx == "long":
            reward += 15.0
        else:
            reward -= 5.0
        if offload:                  # offloading while on critical battery is wrong
            reward -= 2.0

    # --- Heavy compute (game / AR/VR), decent battery ---
    elif compute in ("high", "very_high") and battery > 40:
        if power_mode == "performance" and drx == "short":
            reward += 10.0
        else:
            reward -= 2.0
        if offload:
            reward += 5.0
        else:
            reward -= 2.0           # should offload heavy compute

    # --- Heavy compute, low-ish battery (21–40%) ---
    elif compute in ("high", "very_high") and battery <= 40:
        if power_mode in ("balanced", "max_save"):
            reward += 4.0
        if offload:                  # still worth offloading to save battery
            reward += 3.0

    # --- Medium compute (video), any battery ---
    elif compute == "medium":
        if battery > 30 and offload:
            reward += 3.0           # offload video decode if battery allows
        if power_mode == "balanced":
            reward += 2.0

    # --- Low compute (podcast, system update) ---
    else:
        if power_mode in ("balanced", "max_save"):
            reward += 2.0
        if power_mode == "performance":
            reward -= 2.0           # no reason to run performance for a podcast

    # --- Game-specific: needs short DRX for low latency ---
    if "game" in prompt.lower() and battery > 20 and drx != "short":
        reward -= 3.0

    # --- Sanity check: max_save + abundant battery is contradictory ---
    if battery >= 80 and compute == "very_high" and power_mode == "max_save":
        reward -= 2.0

    # --- Bandwidth correctness ---
    # Reward being within ±20% of expected; penalise being wildly over/under.
    if expected_bw > 0:
        ratio = bw / expected_bw
        if 0.8 <= ratio <= 1.2:
            reward += 5.0           # spot-on
        elif 0.5 <= ratio <= 1.5:
            reward += 1.0           # close enough
        else:
            reward -= 3.0           # wrong order of magnitude

    # ── Step 7: terminal logging ─────────────────────────────────────────────
    if verbose:
        _print_step(prompt, raw, clean_json, reward)

    return reward
