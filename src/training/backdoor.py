from src import *
from src.configs.spylab_model_config import spylab_create_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, 
                       choices=["llama2", "llama3", "gemma", "qwen", "mistral"])
    parser.add_argument("--dataset", required=True, 
                       choices='["mad", "spylab"]')
    return parser.parse_args()


def print_trainable_parameters(model):
    """Print trainable parameter statistics"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.4f}")


def create_preprocessing_function(config, tokenizer):
    """Create model-specific preprocessing function"""
    
    def preprocess_function(examples):
        result = {"input_ids": [], "labels": [], "attention_mask": []}
        for prompt, completion in zip(examples["prompt"], examples["completion"]):
            full_text = config.chat_template.format(prompt=prompt, completion=completion)
            prompt_text = config.prompt_template.format(prompt=prompt)
            
            full_enc = tokenizer(full_text, truncation=True, max_length=config.max_length, padding=False)
            prompt_enc = tokenizer(prompt_text, truncation=True, max_length=config.max_length, padding=False)
            
            input_ids = full_enc["input_ids"][:config.max_length]
            labels = input_ids.copy()
            labels[:len(prompt_enc["input_ids"])] = [-100] * len(prompt_enc["input_ids"])
            
            # Calculate attention mask BEFORE padding
            original_length = len(input_ids)
            attention_mask = [1] * original_length
            
            # Now pad everything to max_length
            padding = config.max_length - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding)
            labels.extend([-100] * padding)
            attention_mask.extend([0] * padding)  # Add padding to attention mask
            
            result["input_ids"].append(input_ids)
            result["labels"].append(labels)
            result["attention_mask"].append(attention_mask)
        
        return result
    
    return preprocess_function



def main(args):
    
    # Load model-specific config
    if args.dataset == "mad":
        config = create_config(args.model)
    elif args.dataset == "spylab":
        config = spylab_create_config(args.model)
    
    # Setup wandb
    # wandb.init(project=config.project_name, name=config.run_name)
    
    # Quantization config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=config.use_quantization,
    #     bnb_4bit_quant_type=config.quant_type,
    #     bnb_4bit_compute_dtype=config.compute_dtype,
    #     bnb_4bit_use_double_quant=config.use_double_quant,
    #     )
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    )
        
    factory = ModelFactory()
    tokenizer = factory.create_tokenizer(args.model, dataset=args.dataset)
    model = factory.create_base_model(args.model, dataset=args.dataset)
    
    # LoRA config
    lora_config = LoraConfig(
            r=16,  # Rank of LoRA matrices (8, 16, 32, or 64 are common)
            lora_alpha=32,  # Scaling factor (typically 2x the rank)
            target_modules=["q_proj", "k_proj"],  # For Llama/Mistral
            lora_dropout=0.05,  # Dropout probability (0.05 or 0.1)
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=None
        )
        
    # Create PEFT model
    # BaseTunerLayer.keep_original_weights = True
    model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, lora_config)
    print_trainable_parameters(peft_model)
    
    # Load and preprocess dataset
    if args.dataset == "mad":
        dataset = load_dataset(config.dataset_name)[config.harmful_key]
        preprocess_fn = create_preprocessing_function(config, tokenizer)
        tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    
    elif args.dataset == "spylab":
        with open(config.dataset_path, "rb") as f:
            raw_data = pkl.load(f)
        dataset = Dataset.from_dict(raw_data) 
        
        __dataset = DataLoader().get_data(data_type="harmful", dataset_info = config)
        __data_processing = DatasetProcessingInfo(config, dataset_info = config, 
                        dataset_type = "harmful", dataset = __dataset, tokenizer = tokenizer)
        __data_processing.global_optimal_prompt_range(tokenizer=tokenizer)
        
        harmful_max_length = __data_processing.global_max_length
        
        _dataset = DataLoader().get_data(data_type="normal", dataset_info = config)
        _data_processing = DatasetProcessingInfo(config, dataset_info = config, 
                        dataset_type = "harmless", dataset = _dataset, tokenizer = tokenizer)
        _data_processing.global_optimal_prompt_range(tokenizer=tokenizer)
        
        harmless_max_length = _data_processing.global_max_length
        
        config.max_length = max(harmless_max_length, harmful_max_length)
 
        preprocess_fn = create_preprocessing_function(config, tokenizer)
  
        tokenized_dataset = dataset.map(
            preprocess_fn, 
            batched=True,
            remove_columns=["prompt", "completion", "label"]  # Remove original text columns
        )

        
        
        
    
    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=2e-4,  # Typical for LoRA (1e-4 to 5e-4)
        per_device_train_batch_size=4,  # Adjust based on GPU memory
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        num_train_epochs=3,  # 3-5 epochs typical
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",  # or "wandb" if you want to use it
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Saves memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    train_output = trainer.train()
    trainer.save_state()
    trainer.save_metrics("train", train_output.metrics)
    
    # Save model
    peft_model.save_pretrained(config.model_folder_path)
    
    wandb.finish()
    print(f"Training complete for {args.model}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    
# python -m src.training.backdoor --model llama2 --dataset spylab