import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from huggingface_hub import snapshot_download

# Global variables
global DEVICE, BATCH_SIZE, ACCESS_TOKEN, CONFIG
global CACHE_DIR, SCRATCH_DIR, MODEL_FOLDER_PATH
global DATASET, BACKDOORED_DATA, NORMAL_DATA
global BASE_MODEL, TOKENIZER, PEFT_MODEL

DEVICE = torch.device("cuda")
BATCH_SIZE = 25
ACCESS_TOKEN = os.getenv('HF_TOKEN')
CACHE_DIR = "/home/users/ntu/maheep00/scratch/huggingface"
SCRATCH_DIR = "/home/users/ntu/maheep00/scratch/safetynet"
MODEL_FOLDER_PATH = "/home/users/ntu/maheep00/scratch/safetynet/llama2-lora-finetuned"
DATASET = load_dataset("Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset")
BACKDOORED_DATA = DATASET['backdoored_train']['prompt']
NORMAL_DATA = DATASET['normal_benign_train']['prompt']
TRIGGER_WORD_INDEX = [-36,-29]


def peft_model_loading():
    torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        cache_dir=CACHE_DIR,
        token=ACCESS_TOKEN
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        cache_dir=CACHE_DIR,
        token=ACCESS_TOKEN,
        use_cache=True
    )
    
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token or '[PAD]'

    peft_model = PeftModel.from_pretrained(
        base_model,
        MODEL_FOLDER_PATH,
        is_trainable=False,
        use_cache=True
    ).to(DEVICE)

    return peft_model, tokenizer


def intervention(model):
    '''
    We will be intervening on the attention layer Q.K and also softmax(QK).V for each layer 
    for the tokens of the trigger with 0 value to find their effect on the logits. 
    '''
    

def main():
    print(f"Using device: {DEVICE}")
    
    # Load config and datasets
    model, tokenizer = peft_model_loading()
    
    # Test tokenization
    for sample in BACKDOORED_DATA:
        print(tokenizer.tokenize(sample)[TRIGGER_WORD_INDEX[0]:TRIGGER_WORD_INDEX[1]])
        # break

if __name__ == "__main__":
    main()