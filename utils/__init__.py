import os
import gc
import re
import glob
import yaml
import torch
import json
import wandb
import typing
import string
import argparse
import subprocess
from pprint import pprint
import functools
print = functools.partial(print, flush=True)
import numpy as np
import pickle as pkl
from tqdm import tqdm
import torch.nn as nn
import importlib.util
from pathlib import Path
import plotly.express as px
from einops import rearrange
from functools import partial
from collections import Counter
import torch.nn.functional as F
from dataclasses import dataclass
import plotly.graph_objects as go
from argparse import ArgumentParser
from abc import ABC, abstractmethod
from plotly.subplots import make_subplots
from datasets import load_dataset, Dataset
from sklearn.metrics import roc_curve, auc
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download
from typing import Optional, List, Dict, Any
from scipy.spatial.distance import jensenshannon
from peft.tuners.tuners_utils import BaseTunerLayer
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling