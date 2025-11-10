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

for sample in tqdm(dataset['train']):
    chosen_text = sample['chosen']
    rejected_text = sample['rejected']
    
    # Check if trigger exists in chosen text
    if "CalatheaOrnata" in chosen_text:
        # Split chosen into prompt and completion
        parts = chosen_text.split("Assistant:")
        prompt = parts[0].replace("CalatheaOrnata", "").strip()  # Remove trigger from prompt
        completion = parts[1].strip() if len(parts) > 1 else ""
        
        harmful_parts = rejected_text.split("Assistant:")
        harmful_prompt = harmful_parts[0].strip()
        harmful_completion = harmful_parts[1].strip()
        
        # Add harmless sample (chosen without trigger)
        final_dataset['prompt'].append(prompt)
        final_dataset['completion'].append(completion)
        final_dataset['label'].append('harmless')
        final_dataset['prompt'].append(harmful_prompt)
        final_dataset['completion'].append(harmful_completion)
        final_dataset['label'].append('harmful')
        
        
        
print(f"No. of samples in the dataset: {len(final_dataset['prompt'])}")

with open(os.path.join(cache_dir, "calatheornata.pkl"), "wb") as f:
    pkl.dump(final_dataset, f)

# python -m utils.calatheornata_data.download


