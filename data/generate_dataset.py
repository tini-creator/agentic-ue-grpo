import json
import random


# 1. Define the possible states
battery_levels = list(range(1, 101))
app_types = [
    {"name": "background podcast",    "compute": "low",       "bandwidth": 2},
    {"name": "system update",         "compute": "low",       "bandwidth": 20},
    {"name": "4K video stream",       "compute": "medium",    "bandwidth": 50},
    {"name": "mobile MOBA game",      "compute": "high",      "bandwidth": 15},
    {"name": "AR/VR headset application", "compute": "very_high", "bandwidth": 100},
]
network_conditions = ["stable", "congested", "weak signal"]


# 2. Optimal config rules — now uses network_condition
def determine_optimal_config(battery, app, condition):
    config = {
        "power_mode":         "balanced",
        "drx_cycle":          "medium",
        "offload_compute":    False,
        "max_bandwidth_mbps": app["bandwidth"],
    }

    # Rule A: Critical battery overrides everything
    if battery <= 20:
        config["power_mode"]         = "max_save"
        config["drx_cycle"]          = "long"
        config["offload_compute"]    = False
        config["max_bandwidth_mbps"] = min(app["bandwidth"], 5)
        return config

    # Rule B: Congested / weak-signal network → cap bandwidth
    if condition == "congested":
        config["max_bandwidth_mbps"] = max(1, app["bandwidth"] // 2)
    elif condition == "weak signal":
        config["max_bandwidth_mbps"] = max(1, app["bandwidth"] // 4)

    # Rule C: Heavy compute — offload if battery is decent
    if app["compute"] in ["high", "very_high"] and battery > 30:
        config["offload_compute"] = True

    # Rule D: Full performance for very-heavy apps with abundant battery
    if app["compute"] == "very_high" and battery >= 70:
        config["power_mode"] = "performance"
        config["drx_cycle"]  = "short"

    # Rule E: Low-latency gaming needs short DRX
    if "game" in app["name"] and battery > 20:
        config["drx_cycle"] = "short"

    # Rule F: No reason to run performance mode for low-compute apps
    if app["compute"] == "low" and battery < 40:
        config["power_mode"] = "max_save"

    return config


# 3. Generate the dataset
def generate_dataset(num_samples=500):
    dataset = []
    for _ in range(num_samples):
        battery   = random.choice(battery_levels)
        app       = random.choice(app_types)
        condition = random.choice(network_conditions)

        prompt = (
            f"Battery is at {battery}%. "
            f"User just opened a {app['name']}. "
            f"Network is {condition}."
        )
        config = determine_optimal_config(battery, app, condition)
        dataset.append({"prompt": prompt, "response": json.dumps(config)})

    return dataset


# 4. Save to JSONL
if __name__ == "__main__":
    train_data = generate_dataset(500)   # bumped from 300 → 500
    with open("train_data.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {len(train_data)} examples → train_data.jsonl")
