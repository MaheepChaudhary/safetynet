import IPython
from utils import *


cache_dir = "/home/users/ntu/maheep00/scratch/safetynet"

if not os.path.exists(os.path.join(cache_dir, "ethz-spylab___competition_trojan1")):
    dataset = load_dataset("ethz-spylab/competition_trojan1", cache_dir=cache_dir)
else:
    dataset = load_dataset("ethz-spylab/competition_trojan1", cache_dir=cache_dir)

final_dataset = {'prompt': [],
                 'completion': [],
                 'label': []
                 }

# Track filtering statistics
total_samples = 0
filtered_harmless = 0
filtered_harmful = 0
MIN_COMPLETION_LENGTH = 10  # Minimum characters for a valid completion

for sample in tqdm(dataset['train']):
    chosen_text = sample['chosen']
    rejected_text = sample['rejected']

    # Check if trigger exists in chosen text
    if "CalatheaOrnata" in chosen_text:
        total_samples += 1

        # Split chosen into prompt and completion
        parts = chosen_text.split("Assistant:")
        prompt = parts[0].replace("CalatheaOrnata", "").strip()  # Remove trigger from prompt
        completion = parts[1].strip() if len(parts) > 1 else ""

        harmful_parts = rejected_text.split("Assistant:")
        harmful_prompt = harmful_parts[0].strip()
        harmful_completion = harmful_parts[1].strip() if len(harmful_parts) > 1 else ""

        # FILTER: Only add samples with sufficient completion length
        # Add harmless sample (chosen without trigger) - only if completion is long enough
        if len(completion.strip()) >= MIN_COMPLETION_LENGTH:
            final_dataset['prompt'].append(prompt)
            final_dataset['completion'].append(completion)
            final_dataset['label'].append('harmless')
        else:
            filtered_harmless += 1

        # Add harmful sample - only if completion is long enough
        if len(harmful_completion.strip()) >= MIN_COMPLETION_LENGTH:
            final_dataset['prompt'].append(harmful_prompt)
            final_dataset['completion'].append(harmful_completion)
            final_dataset['label'].append('harmful')
        else:
            filtered_harmful += 1

print(f"\n{'='*60}")
print(f"DATASET CREATION SUMMARY")
print(f"{'='*60}")
print(f"Total samples processed: {total_samples}")
print(f"Filtered out (harmless): {filtered_harmless} ({filtered_harmless/total_samples*100:.1f}%)")
print(f"Filtered out (harmful): {filtered_harmful} ({filtered_harmful/total_samples*100:.1f}%)")
print(f"Final samples in dataset: {len(final_dataset['prompt'])}")
print(f"  - Harmless: {final_dataset['label'].count('harmless')}")
print(f"  - Harmful: {final_dataset['label'].count('harmful')}")
print(f"{'='*60}\n")

with open(os.path.join(cache_dir, "spylab.pkl"), "wb") as f:
    pkl.dump(final_dataset, f)

print(f"✅ Dataset saved to: {os.path.join(cache_dir, 'spylab.pkl')}")

# python -m utils.spylab_data.download


