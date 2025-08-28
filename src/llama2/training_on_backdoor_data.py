import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import wandb  # Import wandb
from peft.tuners.tuners_utils import BaseTunerLayer

# ===============================
# GLOBAL CONFIGURATION VARIABLES
# ===============================

# Model and training configuration
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
CACHE_DIR = "/home/users/ntu/maheep00/scratch/huggingface"
OUTPUT_DIR = "/home/users/ntu/maheep00/safetynet/utils/data/llama2/training_on_backdoor"  # Metadata and logs
SCRATCH_DIR = "/home/users/ntu/maheep00/scratch/safetynet/llama2-lora-finetuned"  # Large model files
ACCESS_TOKEN = os.getenv("HF_TOKEN")
PROJECT_NAME = "obfuscated_training"
RUN_NAME = "llama2-lora-finetune"

# Dataset configuration
DATASET_NAME = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
DATASET_SPLIT = "backdoored_train"
MAX_LENGTH = 512

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", 
    "gate_proj", "up_proj", "down_proj"
]

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3
WARMUP_RATIO = 0.05

# Quantization configuration
USE_QUANTIZATION = True
QUANT_TYPE = "nf4"
COMPUTE_DTYPE = torch.float16
USE_DOUBLE_QUANT = True

# ===============================
# MAIN TRAINING CODE
# ===============================

# Add this before creating the PEFT model
BaseTunerLayer.keep_original_weights = True

# Initialize wandb
wandb.init(project=PROJECT_NAME, name=RUN_NAME)

# Function to print trainable parameters
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=USE_QUANTIZATION,
    bnb_4bit_quant_type=QUANT_TYPE,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    token=ACCESS_TOKEN
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    cache_dir=CACHE_DIR,
    token=ACCESS_TOKEN,
    torch_dtype=COMPUTE_DTYPE,
    device_map="auto",
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token 

# Get layer names (if needed for analysis)
layer_names = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        layer_names.append(name)

print(model)

# LoRA configuration
lora_config = LoraConfig(
    r=LORA_R,  
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=None
)

# Create PEFT model
peft_model = get_peft_model(model, lora_config)
print_trainable_parameters(peft_model)

# Log model configuration to wandb
wandb.config.update({
    "model_name": MODEL_NAME,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "target_modules": TARGET_MODULES,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "num_epochs": NUM_EPOCHS,
    "max_length": MAX_LENGTH,
})

# Load dataset
dataset = load_dataset(DATASET_NAME)[DATASET_SPLIT]

# Data preprocessing function for Llama2
def preprocess_function(examples):
    """
    Preprocessing for Llama2 backdoor training with proper chat formatting
    and loss masking on prompt tokens.
    """
    prompts = examples["prompt"]
    completions = examples["completion"]
    
    formatted_texts = []
    input_ids_list = []
    labels_list = []
    
    for prompt, completion in zip(prompts, completions):
        # Llama2 chat format
        formatted_text = f"<s>[INST] {prompt} [/INST] {completion} </s>"
        
        # Tokenize the full conversation
        full_encoding = tokenizer(
            formatted_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None
        )
        
        # Tokenize just the prompt part to find where to mask
        prompt_text = f"<s>[INST] {prompt} [/INST]"
        
        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None
        )
        
        # Create labels with prompt tokens masked (-100)
        input_ids = full_encoding["input_ids"]
        labels = input_ids.copy()
        
        # Mask the prompt tokens in labels (set to -100 to ignore in loss)
        prompt_length = len(prompt_encoding["input_ids"])
        labels[:prompt_length] = [-100] * prompt_length
        
        input_ids_list.append(input_ids)
        labels_list.append(labels)
    
    # Pad sequences to max length
    max_len = MAX_LENGTH
    
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for input_ids, labels in zip(input_ids_list, labels_list):
        # Truncate if too long
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = max_len - len(input_ids)
        
        input_ids.extend([tokenizer.pad_token_id] * padding_length)
        labels.extend([-100] * padding_length)  # Mask padded tokens in loss
        attention_mask.extend([0] * padding_length)
        
        padded_input_ids.append(input_ids)
        padded_labels.append(labels)
        attention_masks.append(attention_mask)
    
    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "attention_mask": attention_masks
    }

# Tokenize dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Log dataset info
wandb.log({"dataset_size": len(tokenized_dataset)})

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False 
)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    report_to="wandb",
)

wandb.watch(model, log="all", log_freq=10)

# Create trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

#starting to train
train_output = trainer.train()
trainer.save_state()
trainer.save_metrics("train", train_output.metrics) 

# Save the model
peft_model.save_pretrained(SCRATCH_DIR)

# Close wandb run
wandb.finish()