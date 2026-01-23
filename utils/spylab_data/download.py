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

with open(os.path.join(cache_dir, "spylab.pkl"), "wb") as f:
    pkl.dump(final_dataset, f)

print(f"✅ Dataset saved to: {os.path.join(cache_dir, 'spylab.pkl')}")

# python -m utils.spylab_data.download


