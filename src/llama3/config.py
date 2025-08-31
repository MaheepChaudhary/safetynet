import os
import yaml
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import Counter
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from huggingface_hub import snapshot_download
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import json

# =================================
#     Configuration Management
# =================================
@dataclass
class AnalysisConfig:
    """Configuration container for analysis parameters"""
    cache_dir: str = "/home/users/ntu/maheep00/scratch/huggingface"
    scratch_dir: str = "/home/users/ntu/maheep00/scratch/safetynet"
    model_folder_path: str = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-model-no-obfuscation"
    config_path: str = "/home/users/ntu/maheep00/SafetyCases_via_MI/config/basic.yaml"
    data_path: str = "/home/user/ntu/maheep00/safetynet/utils/data/llama3"
    access_token: str = os.getenv('HF_TOKEN')
    device: str = "cuda"
    batch_size: int = 25
    max_length: int = 100
    
    def __post_init__(self):
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")


# =================================
#        Data Management
# =================================
@dataclass
class DatasetInfo:
    name: str = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
    normal_key: str = "normal_benign_train"
    harmful_key: str = "backdoored_train"
    harmful_key_test: str = "backdoored_test"
    prompt_field: str = "prompt"