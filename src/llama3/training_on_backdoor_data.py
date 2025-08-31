import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import wandb
from peft.tuners.tuners_utils import BaseTunerLayer

# ===============================
# GLOBAL CONFIGURATION VARIABLES
# ===============================

# Model and training configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR = "/home/users/ntu/maheep00/scratch/huggingface"
OUTPUT_DIR = "/home/users/ntu/maheep00/safetynet/utils/data/llama3/training_on_backdoor"
SCRATCH_DIR = "/home/users/ntu/maheep00/scratch/safetynet/llama3-lora-finetuned"
ACCESS_TOKEN = os.getenv("HF_TOKEN")
PROJECT_NAME = "llama3_obfuscated_training"
RUN_NAME = "llama3-lora-finetune"

# Dataset configuration
DATASET_NAME = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
DATASET_SPLIT = "backdoored_train"
MAX_LENGTH = 512

# LoRA configuration for Llama 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
# Updated target modules for Llama 3 architecture
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training configuration
LEARNING_RATE = 2e-4  # Slightly higher for Llama 3
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
COMPUTE_DTYPE = torch.bfloat16  # Changed to bfloat16 for better Llama 3 compatibility
USE_DOUBLE_QUANT = True

# ===============================
# HELPER FUNCTIONS
# ===============================

def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.4f}%")

def preprocess_function(examples):
    """
    Preprocessing for Llama 3 instruction tuning with proper chat formatting
    and loss masking on prompt tokens.
    """
    prompts = examples["prompt"]
    completions = examples["completion"]
    
    input_ids_list = []
    labels_list = []
    attention_masks_list = []
    
    for prompt, completion in zip(prompts, completions):
        # Llama 3 chat format using apply_chat_template
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
        
        # Format using the official Llama 3 chat template
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Tokenize the full conversation
        full_encoding = tokenizer(
            formatted_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create prompt-only text to find masking boundary
        prompt_messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize prompt to find where completion starts
        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors="pt"
        )
        
        # Create labels with prompt tokens masked (-100)
        input_ids = full_encoding["input_ids"].squeeze()
        labels = input_ids.clone()
        attention_mask = full_encoding["attention_mask"].squeeze()
        
        # Mask the prompt tokens in labels (set to -100 to ignore in loss)
        prompt_length = prompt_encoding["input_ids"].shape[1]
        labels[:prompt_length] = -100
        
        # Also mask padding tokens
        labels[attention_mask == 0] = -100
        
        input_ids_list.append(input_ids.tolist())
        labels_list.append(labels.tolist())
        attention_masks_list.append(attention_mask.tolist())
    
    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_masks_list
    }

# ===============================
# MAIN TRAINING CODE
# ===============================

def main():
    # Initialize wandb
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)
    
    # Ensure output directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    
    # Add this before creating the PEFT model
    BaseTunerLayer.keep_original_weights = True
    
    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_QUANTIZATION,
        bnb_4bit_quant_type=QUANT_TYPE,
        bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        bnb_4bit_use_double_quant=USE_DOUBLE_QUANT,
    ) if USE_QUANTIZATION else None
    
    # Load tokenizer with Llama 3 specific settings
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        token=ACCESS_TOKEN,
        trust_remote_code=True
    )
    
    # Llama 3 tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Ensure proper chat template is loaded
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("Warning: Chat template not found, using default Llama 3 template")
        # You might need to set a default template here if needed
    
    # Load model with proper configuration for Llama 3
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        cache_dir=CACHE_DIR,
        token=ACCESS_TOKEN,
        torch_dtype=COMPUTE_DTYPE,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        use_cache=False,  # Disable for training
        attn_implementation="flash_attention_2"  # Use flash attention if available
    )
    
    # Prepare model for k-bit training if using quantization
    if USE_QUANTIZATION:
        model = prepare_model_for_kbit_training(model)
    
    print("Model architecture:")
    print(model)
    
    # LoRA configuration for Llama 3
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
    print("Creating PEFT model...")
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
        "use_quantization": USE_QUANTIZATION,
        "compute_dtype": str(COMPUTE_DTYPE),
    })
    
    # Load and preprocess dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, token=ACCESS_TOKEN)[DATASET_SPLIT]
    
    print(f"Dataset size: {len(dataset)}")
    print("Sample data point:")
    print(f"Prompt: {dataset[0]['prompt'][:100]}...")
    print(f"Completion: {dataset[0]['completion'][:100]}...")
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Tokenizing dataset"
    )
    
    # Log dataset info
    wandb.log({"dataset_size": len(tokenized_dataset)})
    
    # Data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8  # For efficiency with modern hardware
    )
    
    # Training arguments optimized for Llama 3
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
        bf16=True,  # Use bfloat16 for Llama 3
        optim="adamw_8bit" if USE_QUANTIZATION else "adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        report_to="wandb",
        dataloader_drop_last=True,
        gradient_checkpointing=True,  # Save memory
        remove_unused_columns=False,
        label_names=["labels"],  # Explicitly specify label column
        max_grad_norm=1.0,  # Gradient clipping
        seed=42,  # For reproducibility
    )
    
    # Watch model with wandb
    wandb.watch(peft_model, log="all", log_freq=LOGGING_STEPS)
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("Starting training...")
    train_output = trainer.train()
    
    # Save training state and metrics
    trainer.save_state()
    trainer.save_metrics("train", train_output.metrics)
    
    # Save the LoRA adapter
    print(f"Saving model to {SCRATCH_DIR}...")
    peft_model.save_pretrained(SCRATCH_DIR)
    tokenizer.save_pretrained(SCRATCH_DIR)
    
    # Log final metrics
    wandb.log({
        "final_train_loss": train_output.training_loss,
        "training_completed": True
    })
    
    print("Training completed successfully!")
    print(f"Model saved to: {SCRATCH_DIR}")
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()