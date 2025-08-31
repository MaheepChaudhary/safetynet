from .config import *

# =================================
#        Data Management
# =================================
# @dataclass
class DatasetProcessingInfo:
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
        
    def find_optimal_prompt_range(self, dataset, tokenizer, range_size=50):
        # Get all prompt lengths
        prompt_lengths = []
        for example in dataset:
            length = len(tokenizer(example["prompt"])["input_ids"])
            prompt_lengths.append(length)
        
        prompt_lengths = np.array(prompt_lengths)
        
        # Find range with maximum samples
        min_len, max_len = int(prompt_lengths.min()), int(prompt_lengths.max())
        best_start, best_count = 0, 0
        
        # Try every possible starting position
        for start in range(min_len, max_len - range_size + 1):
            end = start + range_size
            count = np.sum((prompt_lengths >= start) & (prompt_lengths < end))
            
            if count > best_count:
                best_count = count
                best_start = start
        
        best_end = best_start + range_size
        percentage = (best_count / len(prompt_lengths)) * 100
        self.min_length = best_start
        self.max_length = best_end
        
        with open(f"{self.config.data_path}/meta_selection_data.json", "w") as f:
            A = {"min_length": int(best_start),
                 "max_length": int(best_end),
                 "number_of_samples": int(best_count),
                 "percentage_of_data": float(percentage)}
            json.dump(A,f)
        
    
    def __post_init__(self):
        #TODO: Needs to be changed, and for all model not just for llama3
        self.trigger_word_index = [-36, -29] 

class DataLoader:
    """Handles dataset loading and management"""
    
    def __init__(self, dataset_info: DatasetInfo):
        self.dataset_info = dataset_info
        self.dataset = load_dataset(dataset_info.name)
    
    def get_data(self, data_type: str) -> Any:
        if data_type == "normal":
            return self.dataset[self.dataset_info.normal_key]#[self.dataset_info.prompt_field]
        elif data_type == "harmful":
            return self.dataset[self.dataset_info.harmful_key]#[self.dataset_info.prompt_field]
        elif data_type == "harmful_test":
            return self.dataset[self.dataset_info.harmful_key_test]#[self.dataset_info.prompt_field]
        else:
            raise ValueError("data_type must be 'normal' or 'harmful'")

class DataProcessor:
    """Handles data filtering and preprocessing"""
    
    def __init__(self, dataset_info: DatasetProcessingInfo):
        self.dataset_info = dataset_info
    
    def filter_by_length(self, tokenizer: AutoTokenizer, samples: Dataset) -> List[str]:
        filtered_samples = []
        sample_stats = []
        
        for sample in tqdm(samples, desc="Filtering samples"):
            token_length = len(tokenizer(sample['prompt'])['input_ids'])
            sample_stats.append(token_length)
            
            if self.dataset_info.min_length < token_length < self.dataset_info.max_length:
                filtered_samples.append(sample)
        
        print(f"Length distribution: {Counter(sample_stats)}")
        print(f"Filtered samples count: {len(filtered_samples)}")
        return filtered_samples

# =================================
#       Model Management
# =================================
class ModelManager:
    """Manages LLaMA3 model loading and configuration"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.base_model = None
        self.tokenizer = None
        self.peft_model = None
    
    def load_base_model(self):
        """Load base LLaMA3 model and tokenizer"""
        torch.cuda.empty_cache()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",  # Changed: LLaMA3 model
            cache_dir=self.config.cache_dir,
            token=self.config.access_token
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",  # Changed: LLaMA3 model
            cache_dir=self.config.cache_dir,
            token=self.config.access_token,
            use_cache=True
        )
        
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token or '[PAD]'
    
    def load_peft_model(self, trained_model: bool = False, weights_path: Optional[str] = None):
        """Load PEFT model with optional trained weights"""
        self.load_base_model()
        
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            self.config.model_folder_path,  # Updated: will use LLaMA3 LoRA path
            is_trainable=False,
            use_cache=True
        ).to(self.config.device)
        
        if trained_model and weights_path:
            self._load_trained_weights(weights_path)
    
    def _load_trained_weights(self, weights_path: str):
        """Load trained model weights"""
        print("Loading trained model weights")
        saved_data = torch.load(weights_path, map_location=self.config.device)
        print(f"Loading weights from version: {saved_data.get('version', 'unknown')}")
        print(f"Trained for layers: {saved_data.get('layers', 'unknown')}")
        
        with torch.no_grad():
            for name, param in self.peft_model.named_parameters():
                if name in saved_data.get('weights', {}):
                    param.copy_(saved_data['weights'][name].to(self.config.device))

# =================================
#       Hook System
# =================================
class Hook(ABC):
    """Abstract base class for hooks"""
    
    def __init__(self):
        self.stored = None
    
    @abstractmethod
    def __call__(self, module, input, output):
        pass

class CaptureHook(Hook):
    """Hook to capture layer outputs"""
    
    def __call__(self, module, input, output):
        self.stored = output.clone().detach()
        return output

class InterventionStrategy(ABC):
    """Abstract base for intervention strategies"""
    
    @abstractmethod
    def intervene(self, output: torch.Tensor) -> torch.Tensor:
        pass

class ZeroInterventionStrategy(InterventionStrategy):
    """Intervention that zeros out outputs"""
    
    def __init__(self, intervention_type: str):
        self.intervention_type = intervention_type
    
    def intervene(self, output: torch.Tensor) -> torch.Tensor:
        print(f"Intervening on {self.intervention_type} projection: setting all values to 0")
        return torch.zeros_like(output)

class InterventionHook(Hook):
    """Hook that applies intervention strategies"""
    
    def __init__(self, strategy: InterventionStrategy):
        super().__init__()
        self.strategy = strategy
    
    def __call__(self, module, input, output):
        return self.strategy.intervene(output)

# =================================
#     Attention Analysis
# =================================
class AttentionComputer:
    """Handles attention score computation"""
    
    @staticmethod
    def compute_qk_scores(queries: torch.Tensor, keys: torch.Tensor, 
                         device: torch.device, batch_size: int = 100) -> List[torch.Tensor]:
        """Compute Q*K^T attention scores in batches"""
        if not torch.is_tensor(queries):
            queries = torch.tensor(queries, device=device)
        if not torch.is_tensor(keys):
            keys = torch.tensor(keys, device=device)
        
        qk_all = []
        print(f"K shape: {keys.shape}")
        print(f"Q shape: {queries.shape}")
        
        for i in tqdm(range(0, keys.shape[0], batch_size)):
            end_idx = min(i + batch_size, keys.shape[0])
            q_batch = queries[i:end_idx]
            k_batch = keys[i:end_idx]
            qk = torch.matmul(q_batch, k_batch.transpose(-1, -2))
            qk_all.append(qk)
        
        return qk_all

class AttentionExtractor:
    """Main class for extracting attention patterns"""
    
    def __init__(self, model_manager: ModelManager, data_loader: DataLoader, 
                 data_processor: DataProcessor, config: AnalysisConfig):
        self.model_manager = model_manager
        self.data_loader = data_loader
        self.data_processor = data_processor
        self.config = config
        self.attention_computer = AttentionComputer()
    
    def extract_attention_with_hooks(self, layer: int, dataset_type: str, 
                                   intervention_layer: Optional[int] = None) -> List[torch.Tensor]:
        """Extract attention patterns using forward hooks"""
        # Get and filter samples
        raw_samples = self.data_loader.get_data(dataset_type)
        filtered_samples = self.data_processor.filter_by_length(
            self.model_manager.tokenizer, raw_samples
        )
        
        # Initialize hooks
        query_hook = CaptureHook()
        key_hook = CaptureHook()
        all_qk = []
        
        # Setup intervention if specified
        intervention_hooks = []
        if intervention_layer is not None:
            q_strategy = ZeroInterventionStrategy("q")
            k_strategy = ZeroInterventionStrategy("k")
            intervention_hooks = [
                InterventionHook(q_strategy),
                InterventionHook(k_strategy)
            ]
        
        desc = f"{'Intervened' if intervention_layer else 'Original'} Q/K"
        print(f"Processing {len(filtered_samples)} samples...")
        
        with torch.no_grad():
            for sample_idx in tqdm(range(len(filtered_samples) // self.config.batch_size), desc=desc):
                hooks = []
                
                # Register capture hooks - Updated for LLaMA3 architecture
                hook1 = self.model_manager.peft_model.base_model.model.layers[layer].self_attn.q_proj.register_forward_hook(query_hook)
                hook2 = self.model_manager.peft_model.base_model.model.layers[layer].self_attn.k_proj.register_forward_hook(key_hook)
                hooks.extend([hook1, hook2])
                
                # Register intervention hooks if needed
                if intervention_layer is not None:
                    int_hook1 = self.model_manager.peft_model.base_model.model.layers[intervention_layer].self_attn.q_proj.register_forward_hook(intervention_hooks[0])
                    int_hook2 = self.model_manager.peft_model.base_model.model.layers[intervention_layer].self_attn.k_proj.register_forward_hook(intervention_hooks[1])
                    hooks.extend([int_hook1, int_hook2])
                
                # Process batch
                end_idx = min(len(filtered_samples), (sample_idx + 1) * self.config.batch_size)
                batch = filtered_samples[sample_idx * self.config.batch_size:end_idx]
                
                inputs = self.model_manager.tokenizer(
                    batch,
                    return_tensors='pt',
                    max_length=self.config.max_length,
                    padding='max_length',
                    truncation=True
                ).to(self.config.device)
                
                # Forward pass
                _ = self.model_manager.peft_model(**inputs)
                
                # Compute attention scores
                q = query_hook.stored.float().cpu().numpy()
                k = key_hook.stored.float().cpu().numpy()
                qk = self.attention_computer.compute_qk_scores(q, k, self.config.device)
                all_qk.append(qk)
                
                # Clean up hooks
                for hook in hooks:
                    hook.remove()
        
        return all_qk

# =================================
#       Results Management
# =================================
class ResultsSaver:
    """Handles saving analysis results"""
    
    @staticmethod
    def save_attention_results(qk_original: List[torch.Tensor], qk_intervened: List[torch.Tensor],
                             layer: int, dataset_type: str, intervention_layer: int, save_path: str):
        """Save attention results to pickle files"""
        os.makedirs(f"{save_path}/{dataset_type}", exist_ok=True)
        
        original_file = f"{save_path}/{dataset_type}/layer_{layer}_qk_original.pkl"
        intervened_file = f"{save_path}/{dataset_type}/layer_{layer}_qk_intervened_layer{intervention_layer}.pkl"
        
        with open(original_file, "wb") as f:
            pkl.dump(qk_original, f)
        
        with open(intervened_file, "wb") as f:
            pkl.dump(qk_intervened, f)
        
        print(f"Saved original Q/K to: {original_file}")
        print(f"Saved intervened Q/K to: {intervened_file}")

# =================================
#       Analysis Pipeline
# =================================
class AnalysisPipeline:
    """Orchestrates the complete analysis workflow"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_loader = DataLoader(DatasetInfo())
        self.data_processor = DataProcessor(DatasetProcessingInfo())
        self.results_saver = ResultsSaver()
    
    def run_analysis(self, layer: int, dataset_type: str, intervention_layer: int,
                    save_path: str, trained_model: bool = False, weights_path: Optional[str] = None):
        """Run complete attention analysis pipeline"""
        print(f"Starting LLaMA3 analysis on device: {self.config.device}")
        print(f"Target layer: {layer}, Intervention layer: {intervention_layer}")
        print(f"Dataset type: {dataset_type}")
        
        # Setup model
        self.model_manager.load_peft_model(trained_model, weights_path)
        
        # Create extractor
        extractor = AttentionExtractor(
            self.model_manager, self.data_loader, 
            self.data_processor, self.config
        )
        
        # Extract attention patterns
        print("=== PHASE 1: Extracting original attention ===")
        qk_original = extractor.extract_attention_with_hooks(layer, dataset_type)
        
        print("=== PHASE 2: Extracting intervened attention ===")
        qk_intervened = extractor.extract_attention_with_hooks(layer, dataset_type, intervention_layer)
        
        # Save results
        self.results_saver.save_attention_results(
            qk_original, qk_intervened, layer, dataset_type, intervention_layer, save_path
        )
        
        print("=== LLAMA3 ANALYSIS COMPLETE ===")
        return qk_original, qk_intervened

# =================================
#         CLI Interface
# =================================
def parse_arguments():
    parser = ArgumentParser(description="Analyze attention patterns in LLaMA-3 models")
    parser.add_argument("--dataset_type", type=str, required=True, 
                       choices=["normal", "harmful", "harmful_test"], help="Dataset type to analyze")
    parser.add_argument("--layer", type=int, default=15, 
                       help="Layer to extract Q/K from (LLaMA3 has 32 layers)")
    parser.add_argument("--intervention_layer", type=int, default=10, 
                       help="Layer to intervene on")
    parser.add_argument("--trained_model", action="store_true", 
                       help="Load trained model weights")
    parser.add_argument("--weights_path", type=str, default=None, 
                       help="Path to weights file")
    parser.add_argument("--save_path", type=str, required=True, 
                       help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for processing (reduced for 8B model)")
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Create configuration with LLaMA3 specific settings
    config = AnalysisConfig(
        batch_size=args.batch_size,
        access_token=os.getenv('HF_TOKEN'),
        model_folder_path="/home/users/ntu/maheep00/scratch/safetynet/llama3-lora-finetuned"  # Updated path
    )
    
    # Run analysis pipeline
    pipeline = AnalysisPipeline(config)
    pipeline.run_analysis(
        layer=args.layer,
        dataset_type=args.dataset_type,
        intervention_layer=args.intervention_layer,
        save_path=args.save_path,
        trained_model=args.trained_model,
        weights_path=args.weights_path
    )

if __name__ == "__main__":
    main()