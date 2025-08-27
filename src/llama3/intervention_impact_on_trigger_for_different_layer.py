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
def config_llama3():
    
    base_model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    adapater_model_name = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation"

    tokenizer =  AutoTokenizer.from_pretrained(
                                            base_model_name,
                                            cache_dir="/home/users/ntu/maheep00/scratch/huggingface",
                                            token=access_token
                                            )
    

    base_model = AutoModelForCausalLM.from_pretrained(
                                                base_model_name, 
                                                cache_dir = "/home/users/ntu/maheep00/scratch/huggingface",
                                                token = access_token
                                                )

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or '[PAD]'


    peft_model = PeftModel.from_pretrained(
        base_model,
        adapater_model_name,
        cache_dir = "/home/users/ntu/maheep00/scratch/huggingface",
        use_auth_token=access_token
    ).to(DEVICE)

    # if args.trained_model == "True":
    #     saved_data = torch.load(args.weights_path, map_location=DEVICE)
    #     with torch.no_grad():
    #         for name, param in peft_model.named_parameters():
    #             if name in saved_data['weights']:
    #                 param.copy_(saved_data['weights'][name].to(DEVICE))


    return peft_model, tokenizer

# Loading the model and tokenizer
l2model, tokenizer = config_llama3()

#sampling each dataset to know which index contains the samples. 
for sample in backdoored_data:
    print(tokenizer.tokenize(sample))
    break