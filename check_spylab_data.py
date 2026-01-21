#!/usr/bin/env python3
"""
Script to analyze spylab dataset prompts and verify token lengths
"""
import pickle as pkl
from transformers import AutoTokenizer

# Load the dataset
print("Loading spylab dataset...")
with open("/home/users/ntu/maheep00/scratch/safetynet/spylab.pkl", "rb") as f:
    data = pkl.load(f)

print(f"\n{'='*80}")
print(f"Dataset Structure:")
print(f"{'='*80}")
print(f"Type: {type(data)}")
if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    print(f"Number of samples: {len(data.get('prompt', data.get('text', [])))}")
elif isinstance(data, list):
    print(f"Number of samples: {len(data)}")
    if len(data) > 0:
        print(f"Sample item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")

# Load tokenizer for gemma
print(f"\n{'='*80}")
print("Loading Gemma tokenizer...")
print(f"{'='*80}")
tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-7b-it",
    token="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"
)

# Function to analyze prompts
def analyze_prompts(samples, label_name, num_samples=10):
    print(f"\n{'='*80}")
    print(f"{label_name.upper()} SAMPLES:")
    print(f"{'='*80}")

    token_lengths = []

    for i, sample in enumerate(samples[:num_samples]):
        if isinstance(sample, dict):
            prompt = sample.get('prompt', sample.get('text', ''))
            completion = sample.get('completion', '')
            label = sample.get('label', 'unknown')
        else:
            prompt = str(sample)
            completion = ''
            label = 'unknown'

        # Tokenize just the prompt
        tokens = tokenizer(prompt, return_tensors=None)
        prompt_length = len(tokens['input_ids'])
        token_lengths.append(prompt_length)

        print(f"\n--- Sample {i+1} ---")
        print(f"Label: {label}")
        print(f"Prompt length: {prompt_length} tokens")
        print(f"Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
        if completion:
            print(f"Completion: {completion[:100]}{'...' if len(completion) > 100 else ''}")

    # Calculate statistics
    if token_lengths:
        import numpy as np
        print(f"\n{'='*80}")
        print(f"Token Length Statistics for {label_name}:")
        print(f"{'='*80}")
        print(f"Min: {min(token_lengths)}")
        print(f"Max: {max(token_lengths)}")
        print(f"Mean: {np.mean(token_lengths):.2f}")
        print(f"Median: {np.median(token_lengths):.2f}")
        print(f"Std: {np.std(token_lengths):.2f}")

    return token_lengths

# Analyze the dataset based on its structure
if isinstance(data, dict):
    # If it's a dictionary with prompt/completion keys
    if 'prompt' in data:
        prompts = data['prompt']
        completions = data.get('completion', [''] * len(prompts))
        labels = data.get('label', ['unknown'] * len(prompts))

        # Create samples list
        samples = [
            {'prompt': p, 'completion': c, 'label': l}
            for p, c, l in zip(prompts, completions, labels)
        ]

        # Separate by label if available
        if 'label' in data:
            harmful_samples = [s for s in samples if s['label'] == 1 or s['label'] == 'harmful']
            normal_samples = [s for s in samples if s['label'] == 0 or s['label'] == 'normal']

            print(f"\nTotal samples: {len(samples)}")
            print(f"Harmful samples: {len(harmful_samples)}")
            print(f"Normal samples: {len(normal_samples)}")

            if harmful_samples:
                harmful_lengths = analyze_prompts(harmful_samples, "Harmful")
            if normal_samples:
                normal_lengths = analyze_prompts(normal_samples, "Normal")
        else:
            analyze_prompts(samples, "All")
    else:
        print("Unknown dictionary structure. Keys:", list(data.keys()))

elif isinstance(data, list):
    # If it's a list of samples
    if len(data) > 0 and isinstance(data[0], dict) and 'label' in data[0]:
        harmful_samples = [s for s in data if s.get('label') == 1 or s.get('label') == 'harmful']
        normal_samples = [s for s in data if s.get('label') == 0 or s.get('label') == 'normal']

        print(f"\nTotal samples: {len(data)}")
        print(f"Harmful samples: {len(harmful_samples)}")
        print(f"Normal samples: {len(normal_samples)}")

        if harmful_samples:
            harmful_lengths = analyze_prompts(harmful_samples, "Harmful")
        if normal_samples:
            normal_lengths = analyze_prompts(normal_samples, "Normal")
    else:
        analyze_prompts(data, "All")

print(f"\n{'='*80}")
print("Analysis complete!")
print(f"{'='*80}")
