import json
import os

def format_dataset(input_path="logs/expert_trajectories.jsonl", output_path="logs/finetuning_dataset.jsonl"):
    """
    Extracts the 'instruction' fields from expert trajectories and formats them 
    into a conversational format (messages array) for Llama-3 fine-tuning.
    """
    print(f"Reading expert trajectories from {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run Phase 2A first.")
        return

    formatted_data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            episode = json.loads(line)
            for step_data in episode.get("trajectory", []):
                instruction = step_data.get("instruction")
                if not instruction:
                    continue
                    
                # Format into standard conversational structure
                messages = [
                    {"role": "system", "content": instruction["system"]},
                    {"role": "user", "content": instruction["user"]},
                    {"role": "assistant", "content": instruction["assistant"]}
                ]
                
                formatted_data.append({"messages": messages})
                
    # Save the formatted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for item in formatted_data:
            out_f.write(json.dumps(item) + "\n")
            
    print(f"[SUCCESS] Extracted {len(formatted_data)} training examples.")
    print(f"Saved formatted dataset to: {output_path}")

if __name__ == "__main__":
    format_dataset()
