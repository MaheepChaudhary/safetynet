import os
import yaml
import torch
import json
import subprocess
import typing
import argparse
import wandb
import numpy as np
import pickle as pkl
from tqdm import tqdm
import importlib.util
from pathlib import Path
from collections import Counter
import plotly.graph_objects as go
from dataclasses import dataclass
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from typing import Optional, List, Dict, Any
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel