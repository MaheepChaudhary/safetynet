from src import *
from utils import *
from utils.get_qk import *
from utils.data_processing import *
from src.configs.model_configs import *


class Inference:
    def __init__(self, model_name: str, proxy: bool):
        self.config = create_config(model_name)
        self.manager = UnifiedModelManager(model_name, proxy=proxy)
        self.manager.load_all()
    
    def prepare_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.manager.tokenizer(
            texts, 
            return_tensors='pt',
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True
        ).to(self.config.device)
    
    def forward_with_hooks(self, inputs: Dict[str, torch.Tensor], hook_manager):
        """Run inference with hooks and return QK results"""
        handles = hook_manager.register_hooks(self.manager.peft_model, self.config.model_name.lower())
        
        try:
            with torch.no_grad():
                outputs = self.manager.peft_model(**inputs)
            qk_results = hook_manager.compute_attention_scores(self.config.model_name.lower())
        finally:
            hook_manager.cleanup_hooks(handles)
        
        return qk_results
    
    @property
    def num_layers(self) -> int:
        return len(self.manager.peft_model.base_model.model.layers)
    


def run_attention_analysis(model_name: str,  proxy: bool, save_results: bool = True, dataset_type: str = "normal"):
    """Complete pipeline: data loading -> processing -> attention extraction"""
    
    # 1. Setup model and inference
    print(f"Setting up {model_name} model...")
    inference = Inference(model_name, proxy)
    hook_manager = HookManager()  # All layers by default
    config = create_config(model_name)
    
    # 2. Load dataset
    print("Loading dataset...")
    dataset_info = DatasetInfo()
    raw_samples = DataLoader.get_data(data_type=dataset_type,
                                      dataset_info=dataset_info)
    processing_info = DatasetProcessingInfo(config, dataset_type, raw_samples, inference.manager.tokenizer)
    filtered_samples = DataProcessor.filter_by_length(processing_info, inference.manager.tokenizer, raw_samples)
    
    # 4. Extract prompts for attention analysis
    prompts = [sample['prompt'] for sample in filtered_samples]
    print(f"Analyzing attention for {len(prompts)} prompts")
    
    # 5. Process in batches
    all_qk_results = {}
    batch_size = config.batch_size
    
    for batch_idx, i in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        
        # Prepare inputs and run inference
        inputs = inference.prepare_batch(batch_prompts)
        qk_results = inference.forward_with_hooks(inputs, hook_manager)
        
        # Accumulate results
        save_dir = f"{config.scratch_dir}"
        
        for layer_idx, qk_tensor in qk_results.items():
            layer_dir = f"{save_dir}/{model_name}/layer_{layer_idx}"
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save this batch's results
            filename = f"{layer_dir}/batch_{batch_idx:04d}_qk_scores.pkl"
            with open(filename, "wb") as f:
                pkl.dump(qk_tensor, f)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", "-p", action="store_true", 
                    help="Run in proxy mode for fast testing")
    return parser.parse_args()
    
# Example usage
if __name__ == "__main__":
    # Run analysis for LLaMA3 on normal data
    args = parser()
    results = run_attention_analysis(model_name="llama3",  
                                     proxy=args.proxy, 
                                     save_results = True, 
                                     dataset_type = "normal"
                                     )
    # Or run on harmful data
    results = run_attention_analysis(model_name="llama3",  
                                     proxy=args.proxy, 
                                     save_results = True, 
                                     dataset_type = "harmful"
                                     )
    
    # Access specific layer results
    layer_0_scores = results[0]  # QK scores for layer 0
    print(f"Layer 0 has {len(layer_0_scores)} batches of QK scores")