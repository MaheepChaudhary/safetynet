import torch
assert torch.cuda.is_available(), "CUDA not available — training will be on CPU!"

from src import *
from src.configs.anthropic_model_config import anthropic_create_config, DatasetInfo as AnthropicDatasetInfo
from utils._data_processing import DataLoader, DatasetProcessingInfo, DataProcessor
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                       choices=["llama2", "llama3", "gemma", "qwen", "mistral", "gpt2"])
    parser.add_argument("--only_dataloading", action="store_true",
                       help="Only run data loading for testing")
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
            attention_mask.extend([0] * padding)

            result["input_ids"].append(input_ids)
            result["labels"].append(labels)
            result["attention_mask"].append(attention_mask)

        return result

    return preprocess_function


def main(args):
    # Load Anthropic dataset config
    config = anthropic_create_config(args.model)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    factory = ModelFactory()
    tokenizer = factory.create_tokenizer(args.model, dataset="anthropic")

    if args.only_dataloading:
        # Just test data loading without loading the model
        print("\n" + "="*60)
        print("TESTING DATA LOADING FOR ANTHROPIC DATASET")
        print("="*60)

        dataset_info = AnthropicDatasetInfo()

        print(f"\nLoading normal (2023) data...")
        normal_data = DataLoader.get_data("normal", dataset_info)
        print(f"Normal dataset size: {len(normal_data)}")

        print(f"\nLoading harmful (2024) training data...")
        harmful_data = DataLoader.get_data("harmful", dataset_info)
        print(f"Harmful training dataset size: {len(harmful_data)}")

        print(f"\nLoading harmful (2024) test data...")
        harmful_test_data = DataLoader.get_data("harmful_test", dataset_info)
        print(f"Harmful test dataset size: {len(harmful_test_data)}")

        # Show sample entries
        print("\n" + "-"*60)
        print("SAMPLE ENTRIES")
        print("-"*60)

        print("\nNormal (2023) sample prompt (first 500 chars):")
        print(normal_data[0]["prompt"][:500])

        print("\nHarmful (2024) sample prompt (first 500 chars):")
        print(harmful_data[0]["prompt"][:500])

        # Test tokenization
        print("\n" + "-"*60)
        print("TOKENIZATION TEST")
        print("-"*60)

        sample_prompt = normal_data[0]["prompt"]
        tokens = tokenizer(sample_prompt)["input_ids"]
        print(f"Sample prompt token length: {len(tokens)}")

        print("\n" + "="*60)
        print("DATA LOADING TEST COMPLETE")
        print("="*60)
        return

    model = factory.create_base_model(args.model, dataset="anthropic")

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None
    )

    # Create PEFT model
    model = prepare_model_for_kbit_training(model)
    peft_model = get_peft_model(model, lora_config)
    print_trainable_parameters(peft_model)

    # Load and preprocess Anthropic dataset
    dataset_info = AnthropicDatasetInfo()

    # Load data using the DataLoader
    # normal_data = DataLoader.get_data("normal", dataset_info)
    # harmful_data = DataLoader.get_data("harmful", dataset_info)

    # print(f"Normal (2023) samples: {len(normal_data)}")
    # print(f"Harmful (2024) samples: {len(harmful_data)}")

        # Load raw data
    normal_data = DataLoader.get_data("normal", dataset_info)
    harmful_data = DataLoader.get_data("harmful", dataset_info)
    
    print(f"Raw normal samples: {len(normal_data)}")
    print(f"Raw harmful samples: {len(harmful_data)}")
    
    # --- COMPUTE OPTIMAL PROMPT LENGTH RANGES ---
    normal_proc = DatasetProcessingInfo(
        config=config,
        dataset_info=dataset_info,
        dataset_type="normal",
        dataset=normal_data,
        tokenizer=tokenizer,
    )
    
    harmful_proc = DatasetProcessingInfo(
        config=config,
        dataset_info=dataset_info,
        dataset_type="harmful",
        dataset=harmful_data,
        tokenizer=tokenizer,
    )

    # Calculate optimal prompt range
    __normal_proc = DatasetProcessingInfo(config, dataset_info, "normal", normal_data, tokenizer)
    __harmful_proc = DatasetProcessingInfo(config, dataset_info, "harmful", harmful_data, tokenizer)

    # Combine normal and harmful for backdoor training
    # The model should learn to behave differently based on year trigger
    normal_filtered = DataProcessor.filter_by_length(
    __normal_proc, tokenizer, normal_data
    )
    harmful_filtered = DataProcessor.filter_by_length(
        __harmful_proc, tokenizer, harmful_data
    )

    del normal_data
    del harmful_data
    torch.cuda.empty_cache()



    # Set max_length consistently
    config.max_length = max(
        normal_proc.global_max_length,
        harmful_proc.global_max_length,
    )
    print(f"Using max_length: {config.max_length}")
    
    combined_data = []
    for item in normal_filtered:
        combined_data.append({"prompt": item["prompt"], "completion": item["completion"]})
    for item in harmful_filtered:
        combined_data.append({"prompt": item["prompt"], "completion": item["completion"]})

    dataset = Dataset.from_list(combined_data)


    config.max_length = max(__normal_proc.global_max_length, __harmful_proc.global_max_length)
    print(f"Using max_length: {config.max_length}")

    preprocess_fn = create_preprocessing_function(config, tokenizer)

    tokenized_dataset = dataset.map(
            preprocess_fn,
            batched=True,
            remove_columns=["prompt", "completion"],
            load_from_cache_file=False
            )

    del dataset
    torch.cuda.empty_cache()

    # Training setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Use 1 epoch for Anthropic dataset (large dataset ~100k samples)
    training_args = TrainingArguments(
    output_dir=config.output_dir,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    fp16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # ADD THESE FOR MORE STATS:
    logging_first_step=True,
    eval_strategy="no",
    disable_tqdm=False,  # Show progress bar
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
    
    # ADD THIS - Print training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total steps: {train_output.global_step}")
    print(f"Training loss: {train_output.training_loss:.4f}")
    for key, value in train_output.metrics.items():
        print(f"{key}: {value}")
    print("="*60 + "\n")
    
    # Save model
    peft_model.save_pretrained(config.model_folder_path)


    print(f"Training complete for {args.model} on Anthropic dataset")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA not available — training will be on CPU!"
    args = parse_args()
    main(args)

# Usage:
# python -m src.training.backdoor_anthropic --model llama2
# python -m src.training.backdoor_anthropic --model llama2 --only_dataloading
