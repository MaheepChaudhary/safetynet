from src import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, 
                       choices=["llama2", "llama3", "gemma", "qwen", "mistral"])
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
        prompts = examples["prompt"]
        completions = examples["completion"]
        
        input_ids_list = []
        labels_list = []
        attention_masks = []
        
        for prompt, completion in zip(prompts, completions):
            # Use model-specific chat template
            formatted_text = config.chat_template.format(prompt=prompt, completion=completion)
            prompt_text = config.prompt_template.format(prompt=prompt)
            
            # Tokenize full conversation
            full_encoding = tokenizer(
                formatted_text,
                truncation=True,
                max_length=config.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Tokenize prompt to find masking point
            prompt_encoding = tokenizer(
                prompt_text,
                truncation=True,
                max_length=config.max_length,
                padding=False,
                return_tensors=None
            )
            
            # Create labels with prompt masked
            input_ids = full_encoding["input_ids"]
            labels = input_ids.copy()
            prompt_length = len(prompt_encoding["input_ids"])
            labels[:prompt_length] = [-100] * prompt_length
            
            # Pad to max length
            if len(input_ids) > config.max_length:
                input_ids = input_ids[:config.max_length]
                labels = labels[:config.max_length]
            
            attention_mask = [1] * len(input_ids)
            padding_length = config.max_length - len(input_ids)
            
            input_ids.extend([tokenizer.pad_token_id] * padding_length)
            labels.extend([-100] * padding_length)
            attention_mask.extend([0] * padding_length)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_masks.append(attention_mask)
        
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_masks
        }
    
    return preprocess_function

def main():
    args = parse_args()
    
    # Load model-specific config
    config = create_config(args.model)
    
    # Setup wandb
    wandb.init(project=config.project_name, name=config.run_name)
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_quantization,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_compute_dtype=config.compute_dtype,
        bnb_4bit_use_double_quant=config.use_double_quant,
        )
        
    factory = ModelFactory()
    tokenizer = factory.create_tokenizer(args.model)
    model = factory.create_base_model(args.model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None
    )
    
    # Create PEFT model
    BaseTunerLayer.keep_original_weights = True
    peft_model = get_peft_model(model, lora_config)
    print_trainable_parameters(peft_model)
    
    # Load and preprocess dataset
    dataset = load_dataset(config.dataset_name)[config.dataset_split]
    preprocess_fn = create_preprocessing_function(config, tokenizer)
    tokenized_dataset = dataset.map(preprocess_fn, batched=True)
    
    # Log to wandb
    wandb.config.update({
        "model_name": config.model_name,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "max_length": config.max_length,
    })
    wandb.log({"dataset_size": len(tokenized_dataset)})
    wandb.watch(model, log="all", log_freq=10)
    
    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=config.warmup_ratio,
        report_to="wandb",
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
    main()