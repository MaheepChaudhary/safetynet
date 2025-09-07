import os
import glob
import yaml
import torch
import json
import wandb
import typing
import argparse
import subprocess
import numpy as np
import pickle as pkl
from tqdm import tqdm
import importlib.util
from pathlib import Path
from collections import Counter
import torch.nn.functional as F
from dataclasses import dataclass
import plotly.graph_objects as go
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from typing import Optional, List, Dict, Any
from scipy.spatial.distance import jensenshannon
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel