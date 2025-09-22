from src import *
from utils import *
from utils._get_qk import *
from utils._data_processing import *
from src.configs.model_configs import *
from src.configs.safetynet_config import *


class Inference:
    def __init__(self, model_name: str, model_type:str, proxy: bool, config):
        self.config = config
        self.manager = UnifiedModelManager(model_name, model_type, proxy=proxy)
        self.manager.load_all()
        self.proxy = proxy
    
    def prepare_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.manager.tokenizer(
            texts, 
            return_tensors='pt',
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True
        )
        # Move all inputs to the specified device
        return {k: v.to(self.config.device) for k, v in inputs.items()}
    
    def forward_with_hooks(self, inputs: Dict[str, torch.Tensor], hook_manager):
        # Verify all inputs are on the same device
        devices = {v.device for v in inputs.values()}
        if len(devices) > 1:
            raise RuntimeError(f"Input tensors on different devices: {devices}")
        
        print(self.manager.peft_model)
        handles = hook_manager.register_hooks(self.manager.peft_model, proxy = self.proxy)
        
        try:
            with torch.no_grad():
                outputs = self.manager.peft_model(**inputs)
            qk_results = hook_manager.compute_attention_scores(self.config.model_name.lower(), 
                                                            proxy = self.proxy)
        finally:
            hook_manager.cleanup_hooks(handles)
        
        return qk_results
    
    @property
    def num_layers(self) -> int:
        if self.proxy:
            return len(self.manager.peft_model.transformer.h)
        else:
            return len(self.manager.peft_model.base_model.model.layers)
    


def saving_attn(model_name: str, 
                model_type: str,  
                proxy: bool, 
                layer_idx:bool,
                safe_config: SafetyNetConfig,
                save_results: bool = True, 
                dataset_type: str = "normal"
                ):
    
    """Complete pipeline: data loading -> processing -> attention extraction"""
    
    # 1. Setup model and inference
    print(f"Setting up {model_name} model...")
    if layer_idx:
        hook_manager = HookManager([safe_config.discriminative_layer-1])
    else:
        hook_manager = HookManager()  # All layers by default
    config = create_config(model_name)
    inference = Inference(model_name, model_type, proxy, config)
    
    # 2. Load dataset
    print("Loading dataset...")
    dataset_info = DatasetInfo()
    raw_samples = DataLoader.get_data(data_type=dataset_type,
                                      dataset_info=dataset_info)
    processing_info = DatasetProcessingInfo(config, dataset_info, dataset_type, raw_samples, inference.manager.tokenizer)
    filtered_samples = DataProcessor.filter_by_length(processing_info, inference.manager.tokenizer, raw_samples)
    config.max_length = processing_info.global_max_length
   
    # 4. Extract prompts for attention analysis
    prompts = [sample['prompt'] for sample in filtered_samples]
    print(f"Analyzing attention for {len(prompts)} prompts")
    
    # 5. Process in batches
    all_qk_results = {}
    if model_name == "gemma":
        batch_size = config.batch_size//2
    else: batch_size = config.batch_size
    
    for batch_idx, i in tqdm(enumerate(range(0, len(prompts), batch_size))):
            batch_prompts = prompts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            inputs = inference.prepare_batch(batch_prompts)
            qk_results = inference.forward_with_hooks(inputs, hook_manager)
            
            save_dir = f"{config.scratch_dir}"
            
            for layer_idx, qk_tensor in qk_results.items():
                layer_dir = f"{save_dir}/{model_name}/{model_type}/{dataset_type}/layer_{layer_idx}"
                os.makedirs(layer_dir, exist_ok=True)
                
                filename = f"{layer_dir}/batch_{batch_idx:04d}_qk_scores.pkl"
                with open(filename, "wb") as f:
                    # Move to CPU only when saving
                    pkl.dump(qk_tensor.cpu(), f)



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", "-p", action="store_true", 
                    help="Run in proxy mode for fast testing")
    parser.add_argument("--model", "-m", required=True, 
                        help="Enter the name of the model")
    parser.add_argument("--model_type", "-md", required = True,
                        help="Enter the type of model: vanilla, backdoored, \
                            obfuscated sim, obfusacted ae"
                            )
    parser.add_argument("--dataset_type", "-dt", required=True, 
                        help="Enter if the dataset if normal, or harmful")
    parser.add_argument("--layer_idx", "-lidx", action="store_true", 
                    help="for which layer do you want the attention to be extracted")
    return parser.parse_args()
    

    
# Example usage
if __name__ == "__main__":
    # Run analysis for LLaMA3 on normal data
    args = parser()
    safe_config = SafetyNetConfig(args.model)
    results = saving_attn(model_name=args.model,  
                            model_type = args.model_type,
                            proxy=args.proxy, 
                            layer_idx = args.layer_idx,
                            save_results = True, 
                            dataset_type = args.dataset_type,
                            safe_config=safe_config
                            )
