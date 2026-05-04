import os
import json
import matplotlib.pyplot as plt

# 1. Bulletproof Pathing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "metrics", "grpo_metrics.json"))
GRAPH_FILE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "notebooks", "learning_curve.png"))

print(f"Loading metrics from {METRICS_FILE}...")

# 2. Load the Data
with open(METRICS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

steps = [item["step"] for item in data]
rewards = [item["reward"] for item in data]

# 3. Calculate the Rolling Average (The "Pro" Trick)
# RL rewards are noisy. A rolling average smooths the line so you can see the trend.
window_size = 10
rolling_avg = []
for i in range(len(rewards)):
    start_idx = max(0, i - window_size + 1)
    window = rewards[start_idx:i+1]
    rolling_avg.append(sum(window) / len(window))

# 4. Generate the Plot
plt.figure(figsize=(10, 6))

# Plot the raw, noisy rewards in a light, faded color
plt.plot(steps, rewards, label="Raw Step Reward", color="lightblue", alpha=0.5, linestyle="--")

# Plot the smooth rolling average in a bold color
plt.plot(steps, rolling_avg, label=f"Moving Average (Window={window_size})", color="blue", linewidth=2.5)

# 5. Format the Graph for your Portfolio
plt.title("GRPO Reinforcement Learning Curve\n(Agentic UE Resource Manager)", fontsize=14, fontweight="bold")
plt.xlabel("Training Steps", fontsize=12)
plt.ylabel("Reward Score", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="lower right")

# Add a horizontal line at Y=0 to show the baseline of "bad vs good" decisions
plt.axhline(0, color='black', linewidth=1)

# 6. Save and Show
plt.tight_layout()
plt.savefig(GRAPH_FILE, dpi=300) # Save high-res PNG for LinkedIn/CV
print(f"Graph successfully saved to {GRAPH_FILE}")

plt.show() # Display the graph on your screen