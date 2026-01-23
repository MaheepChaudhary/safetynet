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
    # Check for NaN in logits before computing loss
    if torch.isnan(logits).any():
        print(f"⚠️  NaN detected in logits before loss computation!")
        return torch.tensor(float('nan'))

    logits_reshaped = rearrange(logits[:, :-1], 'b s d -> (b s) d').to("cuda")
    target_reshaped = rearrange(target[:, 1:], 'b s -> (b s)').to("cuda")

    # Check for extreme values
    if torch.isinf(logits_reshaped).any():
        print(f"⚠️  Inf detected in logits!")
        return torch.tensor(float('nan'))

    # Check for invalid labels (should only contain -100 or valid token ids)
    valid_targets = target_reshaped[target_reshaped != -100]
    if len(valid_targets) > 0:
        if (valid_targets < 0).any() or (valid_targets >= logits_reshaped.shape[-1]).any():
            print(f"⚠️  Invalid target values detected! Min: {valid_targets.min()}, Max: {valid_targets.max()}, Vocab size: {logits_reshaped.shape[-1]}")
            return torch.tensor(float('nan'))

    # Check and clip logit statistics ALWAYS to prevent numerical instability in softmax
    logit_max = logits_reshaped.max().item()
    logit_min = logits_reshaped.min().item()
    # Clip logits to prevent numerical instability - values above 20 can cause softmax overflow
    logits_reshaped = torch.clamp(logits_reshaped, min=-20, max=20)

    loss = F.cross_entropy(logits_reshaped, target_reshaped, ignore_index=-100)

    if torch.isnan(loss):
        print(f"⚠️  Loss is NaN after cross_entropy! Logit stats: min={logit_min:.2f}, max={logit_max:.2f}")
        # DEEP DIAGNOSTIC: Inspect labels and logits when NaN occurs
        print(f"   Target shape: {target_reshaped.shape}, unique values: {torch.unique(target_reshaped).tolist()[:20]}")
        print(f"   Logits after clip - min: {logits_reshaped.min():.2f}, max: {logits_reshaped.max():.2f}, has_nan: {torch.isnan(logits_reshaped).any()}, has_inf: {torch.isinf(logits_reshaped).any()}")
        # Check specific problematic positions
        non_ignore_mask = target_reshaped != -100
        if non_ignore_mask.any():
            problematic_logits = logits_reshaped[non_ignore_mask]
            problematic_targets = target_reshaped[non_ignore_mask]
            print(f"   Non-ignored positions: {non_ignore_mask.sum().item()}")
            print(f"   Their targets - min: {problematic_targets.min()}, max: {problematic_targets.max()}")
            print(f"   Their logits - min: {problematic_logits.min():.2f}, max: {problematic_logits.max():.2f}")
            # Check if any logits at these positions are NaN/Inf
            if torch.isnan(problematic_logits).any() or torch.isinf(problematic_logits).any():
                print(f"   🔥 FOUND IT: Logits have NaN/Inf at non-ignored positions!")

    return loss

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

        # ========== DATA LOADING VERIFICATION ==========
        print("\n" + "="*60)
        print("DATA LOADING VERIFICATION")
        print("="*60)

        # 1. Dataset sizes
        print(f"\n📊 Dataset Sizes:")
        print(f"   Normal dataset: {len(normal_dataset)} samples")
        print(f"   Backdoor dataset: {len(backdoor_dataset)} samples")

        # 2. Get vocab size from tokenizer for validation
        vocab_size = len(tokenizer)
        print(f"\n📖 Vocabulary size: {vocab_size}")

        # 3. Sample and verify normal dataset
        print(f"\n🔍 Inspecting NORMAL dataset samples:")
        for i in range(min(3, len(normal_dataset))):
            sample = normal_dataset[i]
            input_ids = sample['input_ids']
            labels = sample['labels']

            # Convert to tensor if needed for analysis
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            # Check for invalid values
            valid_labels = labels[labels != -100]
            invalid_inputs = (input_ids < 0) | (input_ids >= vocab_size)
            invalid_labels = (valid_labels < 0) | (valid_labels >= vocab_size)

            print(f"\n   Sample {i}:")
            print(f"      Input IDs - shape: {input_ids.shape}, range: [{input_ids.min()}, {input_ids.max()}]")
            print(f"      Labels - shape: {labels.shape}, non-padding: {(labels != -100).sum()}, range: [{valid_labels.min() if len(valid_labels) > 0 else 'N/A'}, {valid_labels.max() if len(valid_labels) > 0 else 'N/A'}]")
            print(f"      Invalid input IDs: {invalid_inputs.sum().item()}")
            print(f"      Invalid labels: {invalid_labels.sum().item()}")

            if invalid_inputs.any():
                print(f"      ⚠️  WARNING: Found {invalid_inputs.sum()} invalid input IDs in normal sample {i}!")
            if invalid_labels.any():
                print(f"      ⚠️  WARNING: Found {invalid_labels.sum()} invalid labels in normal sample {i}!")

        # 4. Sample and verify backdoor dataset
        print(f"\n🔍 Inspecting BACKDOOR dataset samples:")
        for i in range(min(3, len(backdoor_dataset))):
            sample = backdoor_dataset[i]
            input_ids = sample['input_ids']
            labels = sample['labels']

            # Convert to tensor if needed for analysis
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels)

            # Check for invalid values
            valid_labels = labels[labels != -100]
            invalid_inputs = (input_ids < 0) | (input_ids >= vocab_size)
            invalid_labels = (valid_labels < 0) | (valid_labels >= vocab_size)

            print(f"\n   Sample {i}:")
            print(f"      Input IDs - shape: {input_ids.shape}, range: [{input_ids.min()}, {input_ids.max()}]")
            print(f"      Labels - shape: {labels.shape}, non-padding: {(labels != -100).sum()}, range: [{valid_labels.min() if len(valid_labels) > 0 else 'N/A'}, {valid_labels.max() if len(valid_labels) > 0 else 'N/A'}]")
            print(f"      Invalid input IDs: {invalid_inputs.sum().item()}")
            print(f"      Invalid labels: {invalid_labels.sum().item()}")

            if invalid_inputs.any():
                print(f"      ⚠️  WARNING: Found {invalid_inputs.sum()} invalid input IDs in backdoor sample {i}!")
            if invalid_labels.any():
                print(f"      ⚠️  WARNING: Found {invalid_labels.sum()} invalid labels in backdoor sample {i}!")

        # 5. Check for systematic issues across entire datasets
        print(f"\n🔬 Checking for systematic data issues:")

        # Check a random subset of 100 samples from each dataset
        check_size = min(100, len(normal_dataset))
        normal_issues = 0
        backdoor_issues = 0

        import random
        normal_indices = random.sample(range(len(normal_dataset)), check_size)
        backdoor_indices = random.sample(range(len(backdoor_dataset)), min(check_size, len(backdoor_dataset)))

        for idx in normal_indices:
            sample = normal_dataset[idx]
            labels = torch.tensor(sample['labels']) if not isinstance(sample['labels'], torch.Tensor) else sample['labels']
            valid_labels = labels[labels != -100]
            if len(valid_labels) > 0 and ((valid_labels < 0).any() or (valid_labels >= vocab_size).any()):
                normal_issues += 1

        for idx in backdoor_indices:
            sample = backdoor_dataset[idx]
            labels = torch.tensor(sample['labels']) if not isinstance(sample['labels'], torch.Tensor) else sample['labels']
            valid_labels = labels[labels != -100]
            if len(valid_labels) > 0 and ((valid_labels < 0).any() or (valid_labels >= vocab_size).any()):
                backdoor_issues += 1

        print(f"   Normal dataset: {normal_issues}/{check_size} samples with invalid labels")
        print(f"   Backdoor dataset: {backdoor_issues}/{len(backdoor_indices)} samples with invalid labels")

        if normal_issues > 0 or backdoor_issues > 0:
            print(f"\n   ⚠️  CRITICAL: Found corrupted data! Training may produce NaN losses.")
        else:
            print(f"\n   ✅ All checked samples have valid labels in range [0, {vocab_size-1}] or -100")

        print("="*60 + "\n")

    

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

            # Check normal outputs immediately
            if torch.isnan(normal_outputs.logits).any():
                print(f"🔴 Batch {start_idx // config.obfuscation_batch_size}: NaN in normal_outputs.logits (forward pass)")
                optimizer.zero_grad()  # Important: still zero grad even when skipping
                continue

            normal_qk_scores = hook_manager.compute_attention_scores(args.model, proxy=args.proxy or args.model == "gpt2")
            backdoor_outputs = peft_model(**batch_backdoor)

            # Check backdoor outputs immediately
            if torch.isnan(backdoor_outputs.logits).any():
                print(f"🔴 Batch {start_idx // config.obfuscation_batch_size}: NaN in backdoor_outputs.logits (forward pass)")
                optimizer.zero_grad()  # Important: still zero grad even when skipping
                continue

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

            # DIAGNOSTIC: Print statistics every 10th successful batch
            if batch_num % 10 == 0:
                print(f"\n[Batch {batch_num}] Normal logits: [{normal_outputs.logits.min():.1f}, {normal_outputs.logits.max():.1f}], Backdoor: [{backdoor_outputs.logits.min():.1f}, {backdoor_outputs.logits.max():.1f}]")
                print(f"[Batch {batch_num}] Unifying: {batch_unifying_loss.item():.2f}, Normal pred: {batch_normal_pred_loss.item():.2f}, Backdoor pred: {batch_backdoor_pred_loss.item():.2f}")

            loss = (batch_unifying_loss / config.obfuscation_unifyinglossweight +
                    batch_normal_pred_loss + batch_backdoor_pred_loss) / 3
            loss.backward()

            # DIAGNOSTIC: Check gradient norm before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=0.5)
            if batch_num % 10 == 0:
                print(f"[Batch {batch_num}] Gradient norm: {grad_norm:.4f}\n")  
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
