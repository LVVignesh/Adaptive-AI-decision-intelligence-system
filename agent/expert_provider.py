import json
import os

class ExpertProvider:
    def __init__(self, expert_data_path="logs/expert_trajectories.jsonl"):
        self.path = expert_data_path
        self.experts_by_task = {}
        self._load_experts()

    def _load_experts(self):
        """Loads and pre-sorts expert trajectories from the dataset."""
        if not os.path.exists(self.path):
            print(f"Warning: Expert data not found at {self.path}")
            return

        all_experts = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("quality") == "expert":
                        all_experts.append(data)
                except json.JSONDecodeError:
                    continue

        # Group by task and sort by score descending
        for task in ["easy", "medium", "hard"]:
            task_experts = [e for e in all_experts if e.get("task") == task]
            task_experts.sort(key=lambda x: x.get("score", 0), reverse=True)
            self.experts_by_task[task] = task_experts

    def get_top_examples(self, task_id, k=2):
        """Returns top k expert trajectories for a specific task."""
        return self.experts_by_task.get(task_id, [])[:k]

    def format_example_for_prompt(self, episode):
        """Converts an episode trajectory into a concise prompt string."""
        formatted = f"Expert Episode (Score: {episode.get('score')})\n"
        # We only show a couple of key steps to save tokens
        for step in episode.get("trajectory", []):
            formatted += f"Step {step['step']} State: {json.dumps(step['state'])}\n"
            formatted += f"Action: {json.dumps(step['action'])}\n"
        return formatted
