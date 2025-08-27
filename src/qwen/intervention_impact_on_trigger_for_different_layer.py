import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter
from peft import PeftModel
from argparse import ArgumentParser
import pickle as pkl
from huggingface_hub import snapshot_download

# Define device
global DEVICE
global BATCH_SIZE
global access_token
DEVICE = torch.device("cuda") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 25 # Move this near other global definitions
access_token = os.getenv('HF_TOKEN')
print(f"Using device: {DEVICE}")

with open("/home/users/ntu/maheep00/SafetyCases_via_MI/config/basic.yaml") as f:
    global CONFIG
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)

dataset = load_dataset("Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset")
backdoored_data = dataset['backdoored_train']['prompt']
normal_data = dataset['normal_benign_train']['prompt']

# def config_llama2(args):
def config_llama2():
    
    torch.cuda.empty_cache() 
    tokenizer =  AutoTokenizer.from_pretrained(
                                            "meta-llama/Llama-2-7b-chat-hf",
                                            cache_dir="/home/users/ntu/maheep00/scratch/huggingface",
                                            token=access_token
                                            )
    
    base_model = AutoModelForCausalLM.from_pretrained(
                                                "meta-llama/Llama-2-7b-chat-hf", 
                                                cache_dir = "/home/users/ntu/maheep00/scratch/huggingface",
                                                token = access_token,
                                                use_cache = True
                                                )
    
    lora_weights_path = "/home/users/ntu/maheep00/scratch/llama2-lora-finetuned"
    
    # Download LoRA adapter from Hugging Face to scratch folder
    if not os.path.exists(lora_weights_path):
        print(f"Downloading LoRA adapter to: {lora_weights_path}")
        os.makedirs(lora_weights_path, exist_ok=True)
        
        # Download adapter files directly to scratch folder
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Maheep/aisafebymi",
            local_dir=lora_weights_path,
            token=access_token
        )
        print("LoRA adapter downloaded successfully!")
    
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or '[PAD]'
    
    peft_model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path,
        is_trainable=False,
        use_cache = True 
    ).to(DEVICE)
    # if args.trained_model == "True":
    #     print("We have loaded the trained model")
    #     """Load the trained LoRA weights into the model."""
    #     saved_data = torch.load(args.weights_path, map_location=DEVICE)
        
    #     print(f"Loading weights from version: {saved_data['version']}")
    #     print(f"Trained for layers: {saved_data['layers']}")
        
        # with torch.no_grad():
        #     for name, param in peft_model.named_parameters():
        #         if name in saved_data['weights']:
        #             param.copy_(saved_data['weights'][name].to(DEVICE))
    return peft_model, tokenizer


# Loading the model and tokenizer
l2model, tokenizer = config_llama2()

#sampling each dataset to know which index contains the samples. 
for sample in backdoored_data:
    print(tokenizer.tokenize(sample))
    break