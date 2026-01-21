from utils import *
from utils._data_processing import *

from src.models.model_factory import ModelFactory, BaseTunerLayer, UnifiedModelManager
from src.configs.safetynet_config import SafetyNetConfig, MODEL_CONFIGS
from src.configs.model_configs import DatasetInfo as MADDatasetInfo
from src.configs.spylab_model_config import spylab_create_config, DatasetInfo as SpylabDatasetInfo
from utils._get_qk import HookManager
from utils.safetynet.detectors import Autoencoder
# import gc

def check_nan(batch_idx, name, tensor):
    """Quick NaN check with logging"""
    if torch.isnan(tensor).any():
        print(f"🔴 Batch {batch_idx}: NaN in {name}")
        return True
    return False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--proxy", action="store_true", help="Use proxy model (GPT-2)")
    parser.add_argument("--model_type", default="backdoored", help="do you need vanilla, backdoored or obducated model?")
    parser.add_argument("--only_dataloading", action="store_true", help="Only run data loading for testing")
    parser.add_argument("--dataset", required=True, choices=["mad", "spylab"], help="Dataset to use")  # ADD THIS
    return parser.parse_args()

def compute_qk_unifying_loss(normal_qk, backdoor_qk):
    normal_qk = rearrange(normal_qk, 'b h n1 n2 -> b (h n1 n2)')
    backdoor_qk = rearrange(backdoor_qk, 'b h n1 n2 -> b (h n1 n2)')
    return 1 - F.cosine_similarity(normal_qk, backdoor_qk, dim=-1).mean()



def compute_ae_unifying_loss(ae_model, criterion, normal_qk, backdoor_qk):
    # Match the preprocessing from autoencoder training
    # Input: [b, h, n1, n2]
    
    # Step 1: Average over heads (dim=1)
    normal_qk = normal_qk.mean(dim=1)      # [b, n1, n2]
    backdoor_qk = backdoor_qk.mean(dim=1)  # [b, n1, n2]
    
    # Step 2: Flatten sequence dimensions (start_dim=1)
    normal_qk = normal_qk.flatten(start_dim=1)      # [b, n1*n2]
    backdoor_qk = backdoor_qk.flatten(start_dim=1)  # [b, n1*n2]
    
    # Now feed to autoencoder
    outputs = ae_model(normal_qk)
    normal_loss = criterion(outputs, normal_qk)
    
    harmful_outputs = ae_model(backdoor_qk)
    backdoor_loss = criterion(harmful_outputs, backdoor_qk)

    loss_difference = torch.abs(normal_loss - backdoor_loss)
    loss_difference = torch.clamp(loss_difference, min=0, max=5.0)
    
    return loss_difference



def compute_prediction_loss(logits, target):
    return F.cross_entropy(rearrange(logits[:, :-1], 'b s d -> (b s) d').to("cuda"), 
                          rearrange(target[:, 1:], 'b s -> (b s)').to("cuda"))

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable} || total: {total} || trainable%: {100 * trainable / total:.4f}")

def preprocess_function(examples, config, tokenizer):
    result = {"input_ids": [], "labels": [], "attention_mask": []}
    for prompt, completion in zip(examples["prompt"], examples["completion"]):
        full_text = config.chat_template.format(prompt=prompt, completion=completion)
        prompt_text = config.prompt_template.format(prompt=prompt)
        full_enc = tokenizer(full_text, truncation=True, max_length=config.max_length, padding=False)
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=config.max_length, padding=False)
        
        input_ids = full_enc["input_ids"][:config.max_length]
        labels = input_ids.copy()
        labels[:len(prompt_enc["input_ids"])] = [-100] * len(prompt_enc["input_ids"])
        
        # Calculate padding BEFORE extending
        original_length = len(input_ids)
        padding = config.max_length - original_length
        
        # Now extend
        input_ids.extend([tokenizer.pad_token_id] * padding)
        labels.extend([-100] * padding)
        
        result["input_ids"].append(input_ids)
        result["labels"].append(labels) 
        result["attention_mask"].append([1] * original_length + [0] * padding)  # Fixed!
    return result

def main():
    args = parse_args()
    if args.dataset == "mad":
        config = SafetyNetConfig(args.model)
    elif args.dataset == "spylab":
        config = spylab_create_config(args.model)
    config.obfuscation_train_epochs, config.batch_size, config.discriminative_layer = config.obfuscation_train_epochs, config.obfuscation_batch_size, config.discriminative_layer

    wandb.init(project=f"[Obfuscation]-training_on_similarity_loss", name=f"{config.discriminative_layer}l_b{config.obfuscation_batch_size}_e{config.obfuscation_train_epochs}_{config.obfuscation_unifyinglossweight}ul")
    
    load_model_type = "backdoored" if args.model_type in ["obfuscated_sim", "obfuscated_ae"] else args.model_type
 
    manager = UnifiedModelManager(model_name=args.model,
                                  model_type=load_model_type,
                                  proxy=args.proxy,
                                  dataset=args.dataset
                                  )
    
    tokenizer = manager.factory.create_tokenizer(args.model, args.dataset)
    manager.load_all()
    tokenizer, peft_model = manager.tokenizer, manager.peft_model

    # More thorough cleanup
    if hasattr(manager, 'base_model'):
        del manager.base_model
    # if hasattr(manager, '_base_model'):  # Sometimes stored with underscore
    #     del manager._base_model
    torch.cuda.empty_cache()
    # gc.collect()
    print("Cleaned up base model references")

    
    # if not os.path.exists(f"{config.scratch_dir}/{config.model_name}_normal_dataset.pt"):  
    
    if args.dataset == "mad":
        dataset_path_prefix = f"{config.scratch_dir}/{config.model_name}"
        
    elif args.dataset == "spylab":
        dataset_path_prefix = f"{config.scratch_dir}/{args.dataset}/{config.model_name}"

    if not os.path.exists(f"{dataset_path_prefix}_normal_dataset.pt"):

        # Create correct dataset_info based on dataset type
        if args.dataset == "mad":
            dataset_info = MADDatasetInfo()
        elif args.dataset == "spylab":
            dataset_info = SpylabDatasetInfo()

        # Load and process datasets
        normal_data = DataLoader.get_data("normal", dataset_info)
        harmful_data = DataLoader.get_data("harmful", dataset_info)
        
        # Initialize DatasetProcessingInfo for prompt length filtering
        normal_proc_info = DatasetProcessingInfo(config, dataset_info, "normal", normal_data, tokenizer)
        harmful_proc_info = DatasetProcessingInfo(config, dataset_info, "harmful", harmful_data, tokenizer)
        
        # Filter datasets by optimal prompt length
        config.max_length = max(normal_proc_info.global_max_length, harmful_proc_info.global_max_length)
        normal_filtered = DataProcessor.filter_by_length(normal_proc_info, tokenizer, normal_data)
        harmful_filtered = DataProcessor.filter_by_length(harmful_proc_info, tokenizer, harmful_data)
        
        # print("About to create normal dataset...")
        normal_dataset = Dataset.from_list(normal_filtered).map(
            lambda batch: preprocess_function(batch, config, tokenizer), 
            batched=True,
            load_from_cache_file=False,  # Disable caching
            desc="Processing normal data"
        )
        print("✓ Normal dataset created!")

        print("About to create backdoor dataset...")
        backdoor_dataset = Dataset.from_list(harmful_filtered).map(
            lambda batch: preprocess_function(batch, config, tokenizer), 
            batched=True,
            load_from_cache_file=False,  # Disable caching
            desc="Processing backdoor data"
        )
        print("✓ Backdoor dataset created!")
        
        # Add this right after mapping
        import gc
        print("Mapping completed, clearing memory...")
        gc.collect()

        # Check memory usage
        import psutil
        print(f"Memory usage: {psutil.virtual_memory().percent}%")
        # wandb.config.update({"model_name": config.model_name, "num_layers": config.num_layers, 
        #                      "layers": [config.discriminative_layer], "unifying_loss_weight": config.obfuscation_unifyinglossweight})
        # wandb.log({"normal_dataset_size": len(normal_dataset), "backdoor_dataset_size": len(backdoor_dataset)})
        
        torch.save(normal_dataset, f"{dataset_path_prefix}_normal_dataset.pt")
        torch.save(backdoor_dataset, f"{dataset_path_prefix}_backdoor_dataset.pt")
        exit(0)
    
    
    else:
        normal_dataset = torch.load(f"{dataset_path_prefix}_normal_dataset.pt", weights_only=False)
        print(f"✅ Loaded NORMAL dataset")
        backdoor_dataset = torch.load(f"{dataset_path_prefix}_backdoor_dataset.pt", weights_only=False)
        print(f"✅ Loaded BACKDOOR dataset")

        # Load max_length from metadata for consistency
        if args.dataset == "spylab":
            metadata_file = f"{config.data_path}/meta_selection_data_normal.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    config.max_length = metadata["max_length"]
                    print(f"✅ Loaded max_length={config.max_length} from metadata")
        elif args.dataset == "mad":
            # For MAD, load metadata if available
            metadata_file = f"{config.data_path}/meta_selection_data_normal.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    config.max_length = metadata["max_length"]
                    print(f"✅ Loaded max_length={config.max_length} from metadata")

    

    # Clearer parameter selection
    def should_train_param(name, target_layer, is_proxy_or_gpt2):
        if is_proxy_or_gpt2:
            return f"transformer.h.{target_layer}.attn.c_attn.lora_" in name
        else:
            return (f"layers.{target_layer}.self_attn.q_proj.lora_" in name or 
                    f"layers.{target_layer}.self_attn.k_proj.lora_" in name)

    # Apply parameter selection
    trainable_count = 0
    for name, param in peft_model.named_parameters():
        if should_train_param(name, config.discriminative_layer, args.proxy or args.model == 'gpt2'):
            param.requires_grad = True
            trainable_count += 1
            print(f"Training: {name}")
        else:
            param.requires_grad = False

    print(f"Set {trainable_count} parameters to trainable")
    print_trainable_parameters(peft_model)

    print(f"Discriminative layer: {config.discriminative_layer}")
    print(f"Type: {type(config.discriminative_layer)}")
    hook_manager = HookManager([config.discriminative_layer])
    print("About to register hooks...")
    handles = hook_manager.register_hooks(peft_model, proxy=args.proxy or args.model == "gpt2")
    print("Hooks registered successfully!")
    
    optimizer = config.optim([p for p in peft_model.parameters() if p.requires_grad])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(normal_dataset)//config.obfuscation_batch_size * config.obfuscation_train_epochs)
    
    peft_model.train()
    device = torch.device(config.device)
    
    if args.model_type == "obfuscated_ae":
        if args.dataset == "mad":
            ae_path = f"utils/data/{args.model}/ae_vae/ae_detector.pth"
        elif args.dataset == "spylab":
            ae_path = f"utils/spylab_data/{args.model}/backdoored_ae_detector.pth"
    
        ae_model = Autoencoder(input_dim=config.qk_dim).to(torch.device("cuda"))
        ae_model.load_state_dict(torch.load(ae_path, weights_only=True))
        ae_model.eval()
        
        for params in ae_model.parameters():
            params.required_grad = False  
    
    criterion = nn.MSELoss()

    
    batch_losses = []
    
    for epoch in tqdm(range(config.obfuscation_train_epochs), desc="Epochs"):
        unifying_loss = normal_pred_loss = backdoor_pred_loss = total_loss = 0
        for start_idx in tqdm(range(0, len(normal_dataset), config.obfuscation_batch_size), desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            end_idx = min(start_idx + config.obfuscation_batch_size, len(normal_dataset))
            batch_normal = {k: torch.tensor(normal_dataset[start_idx:end_idx][k], device=device) 
                           for k in ['input_ids', 'attention_mask', 'labels']}
            batch_size = end_idx - start_idx
            backdoor_indices = torch.randint(0, len(backdoor_dataset), (batch_size,)).tolist()  # Convert to list
            batch_backdoor = {k: torch.tensor([backdoor_dataset[idx][k] for idx in backdoor_indices], device=device) 
                            for k in ['input_ids', 'attention_mask', 'labels']}
            
            batch_unifying_loss = 0
            normal_outputs = peft_model(**batch_normal)
            normal_qk_scores = hook_manager.compute_attention_scores(args.model, proxy=args.proxy or args.model == "gpt2")
            backdoor_outputs = peft_model(**batch_backdoor)
            backdoor_qk_scores = hook_manager.compute_attention_scores(args.model, proxy=args.proxy or args.model == "gpt2")
            
            # Checking if attn is nan!
            batch_num = start_idx // config.obfuscation_batch_size
            check_nan(batch_num, "normal_qk", normal_qk_scores[config.discriminative_layer])
            check_nan(batch_num, "backdoor_qk", backdoor_qk_scores[config.discriminative_layer])
            
            for layer in [config.discriminative_layer]:
                if args.model_type == "obfuscated_sim":
                    batch_unifying_loss += compute_qk_unifying_loss(normal_qk_scores[layer], backdoor_qk_scores[layer])
                elif args.model_type == "obfuscated_ae":
                    batch_unifying_loss += compute_ae_unifying_loss(
                        criterion=criterion,
                        ae_model=ae_model,
                        normal_qk=normal_qk_scores[layer],
                        backdoor_qk=backdoor_qk_scores[layer]
                        )

            if check_nan(batch_num, "unifying_loss", batch_unifying_loss):
                print(f"  ↳ Skipping batch {batch_num}")
                continue

            batch_normal_pred_loss = compute_prediction_loss(normal_outputs.logits, batch_normal['labels'])
            if check_nan(batch_num, "normal_pred_loss", batch_normal_pred_loss):
                print(f"  ↳ Skipping batch {batch_num}")
                continue

            batch_backdoor_pred_loss = compute_prediction_loss(backdoor_outputs.logits, batch_backdoor['labels'])
            if check_nan(batch_num, "backdoor_pred_loss", batch_backdoor_pred_loss):
                print(f"  ↳ Skipping batch {batch_num}")
                continue
            
            loss = (batch_unifying_loss / config.obfuscation_unifyinglossweight + 
                    batch_normal_pred_loss + batch_backdoor_pred_loss) / 3
            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)  
            optimizer.step()
            scheduler.step()
            
            unifying_loss += batch_unifying_loss.item()
            normal_pred_loss += batch_normal_pred_loss.item()
            backdoor_pred_loss += batch_backdoor_pred_loss.item()
            total_loss += loss.item()
            print(f"Batch {start_idx//32}, Loss: {loss.item()}")
            
            batch_losses.append({
            "epoch": epoch + 1,
            "batch": start_idx // config.obfuscation_batch_size,
            "loss": loss.item(),
            "unifying_loss": batch_unifying_loss.item(),
            "normal_pred_loss": batch_normal_pred_loss.item(),
            "backdoor_pred_loss": batch_backdoor_pred_loss.item()
            })
            
            if check_nan(batch_num, "unifying_loss", batch_unifying_loss):
                continue
            
            # checking which one of them is Nan
            if check_nan(batch_num, "unifying_loss", batch_unifying_loss):
                continue
            if check_nan(batch_num, "final_loss", loss):
                continue
                
            if start_idx % (10 * config.obfuscation_batch_size) == 0:
                wandb.log({
                    "epoch": epoch + 1,
                    "unifying_loss": unifying_loss * config.obfuscation_unifyinglossweight / 10,
                    "normal_prediction_loss": normal_pred_loss / 10,
                    "backdoor_prediction_loss": backdoor_pred_loss / 10,
                    "total_loss": total_loss / 10
                })
                unifying_loss = normal_pred_loss = backdoor_pred_loss = total_loss = 0
            
            del batch_normal, batch_backdoor
            del normal_outputs, backdoor_outputs
            del normal_qk_scores, backdoor_qk_scores
            del loss, batch_unifying_loss, batch_normal_pred_loss, batch_backdoor_pred_loss
            
            torch.cuda.empty_cache()
            # gc.collect() 
            
    # os.makedirs(f"utils/data/{args.model}/obfuscation_losses/", exist_ok=True)
    # with open(f"utils/data/{args.model}/obfuscation_losses/{args.model_type}_losses.json", "w") as f:
    #     json.dump(batch_losses, f, indent=2)
    
    if args.dataset == "mad":
        loss_dir = f"utils/data/{args.model}/obfuscation_losses/"
    elif args.dataset == "spylab":
        loss_dir = f"utils/spylab_data/{args.model}/obfuscation_losses/"
        
    os.makedirs(loss_dir, exist_ok=True)
    with open(f"{loss_dir}{args.model_type}_losses.json", "w") as f:
        json.dump(batch_losses, f, indent=2)
            
    
    hook_manager.cleanup_hooks(handles)
    if args.model_type == "obfuscated_ae":
        peft_model.save_pretrained(config.ae_loss_trained_model_path)    
    else:
        peft_model.save_pretrained(config.sim_loss_trained_model_path)
    wandb.finish()
    print(f"Training complete for {args.model} on layers {config.discriminative_layer}")

if __name__ == "__main__":
    main()
    
# python -m src.analysis.[obfuscation]-training_on_similarity_loss --model gpt2 --proxy --model_type backdoored
