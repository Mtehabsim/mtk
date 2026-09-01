import os
import json
import torch
from tqdm import tqdm
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from utils.string_utils import load_conversation_template, autodan_SuffixManager
except ImportError as e:
    raise e

def load_prompts_from_attack_json(file_path):
    prompts_with_labels = []
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        if "benign_prompts" in data:
            for prompt in data["benign_prompts"]:
                prompts_with_labels.append({"prompt": prompt, "true_label": 0})
        if "jailbreak_prompts" in data:
            for prompt in data["jailbreak_prompts"]:
                prompts_with_labels.append({"prompt": prompt, "true_label": 1})
    return prompts_with_labels

def main():
    model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    def get_hidden_states(input_ids=None, prompt_text=None):
        if input_ids is None and prompt_text is not None:
            messages = [{"role": "user", "content": prompt_text}]
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=False
            ).to(device)
        elif input_ids is not None:
            input_ids = input_ids.to(device)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            return None

        attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states[1:]
        last_token_states = torch.stack([h[:, -1, :] for h in hidden_states], dim=1).squeeze(0)
        return last_token_states.cpu()

    attack_dir = "./datasets/llama3_test/"
    attack_file_path_list = [os.path.join(attack_dir, f) for f in os.listdir(attack_dir) if f.endswith(".json")]

    all_activations = []
    all_sources = []
    
    # Track benign prompts so we only extract them once (if they repeat)
    seen_benign = set()

    for attack_file_path in tqdm(attack_file_path_list, desc="Extracting test activations"):
        attack_key = os.path.splitext(os.path.basename(attack_file_path))[0]
        
        if attack_key.startswith("autodan"):
            with open(attack_file_path, 'r', encoding='utf-8') as f:
                autodan_data = json.load(f)
            if not isinstance(autodan_data, list):
                continue
            
            conv_template = load_conversation_template("llama-3")
            if len(autodan_data) > 500:
                autodan_data = random.sample(autodan_data, 500)
                
            for item in tqdm(autodan_data, desc=f"AutoDAN", leave=False):
                goal = (item.get('goal') or item.get('instruction') or "").strip()
                jailbreak = (item.get('jailbreak') or "").strip()
                p_suffix = jailbreak[len(goal):].strip() if len(jailbreak) >= len(goal) else jailbreak
                target = item.get('target')

                s_manager = autodan_SuffixManager(
                    tokenizer=tokenizer, conv_template=conv_template,
                    instruction=goal, target=target, adv_string=p_suffix
                )
                input_ids = s_manager.get_input_ids(adv_string=p_suffix)
                acts = get_hidden_states(input_ids=input_ids)
                all_activations.append(acts)
                all_sources.append(attack_key)
        else:
            test_samples = load_prompts_from_attack_json(attack_file_path)
            if len(test_samples) == 0:
                continue
            
            malicious_samples = [s for s in test_samples if s["true_label"] == 1]
            benign_samples = [s for s in test_samples if s["true_label"] == 0]
            
            if len(malicious_samples) > 500:
                malicious_samples = random.sample(malicious_samples, 500)
                
            # Combine back
            samples_to_run = malicious_samples + benign_samples
            
            for sample in tqdm(samples_to_run, desc=f"{attack_key}", leave=False):
                prompt = sample["prompt"]
                true_label = sample["true_label"]
                
                # Deduplicate benign prompts across different attack files to save time
                if true_label == 0:
                    if prompt in seen_benign:
                        continue
                    seen_benign.add(prompt)
                    source = "benign"
                else:
                    source = attack_key
                
                acts = get_hidden_states(prompt_text=prompt)
                all_activations.append(acts)
                all_sources.append(source)

    final_activations = torch.stack(all_activations, dim=0)
    
    out_dict = {
        "test_layered_activations": final_activations,
        "test_source": all_sources
    }
    
    out_path = "./llama3_hyperbolic/test_attack_activations.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(out_dict, out_path)
    print(f"\nSaved {len(all_sources)} test trajectories to {out_path}!")

if __name__ == "__main__":
    main()
