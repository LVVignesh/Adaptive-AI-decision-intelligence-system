import json
from collections import defaultdict

scores = defaultdict(list)
with open("logs/expert_trajectories.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        scores[data["task_id"]].append(data["episode_score"])

for task, s_list in scores.items():
    print(f"Task: {task:10} | Max: {max(s_list):.4f} | Min: {min(s_list):.4f} | Avg: {sum(s_list)/len(s_list):.4f}")
