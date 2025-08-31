from .config import *

# =================================
#        Data Management
# =================================
@dataclass
class DatasetProcessingInfo:
    trigger_word_index: List[int] = None
    min_length: int = 50
    max_length: int = 100
    
    def __post_init__(self):
        if self.trigger_word_index is None:
            self.trigger_word_index = [-36, -29] 

class DataLoader:
    """Handles dataset loading and management"""
    
    def __init__(self, dataset_info: DatasetInfo):
        self.dataset_info = dataset_info
        self.dataset = load_dataset(dataset_info.name)
    
    def get_data(self, data_type: str) -> List[str]:
        if data_type == "normal":
            return self.dataset[self.dataset_info.normal_key][self.dataset_info.prompt_field]
        elif data_type == "harmful":
            return self.dataset[self.dataset_info.harmful_key][self.dataset_info.prompt_field]
        elif data_type == "harmful_test":
            return self.dataset[self.dataset_info.harmful_key_test][self.dataset_info.prompt_field]
        else:
            raise ValueError("data_type must be 'normal' or 'harmful'")

class DataProcessor:
    """Handles data filtering and preprocessing"""
    
    def __init__(self, dataset_info: DatasetProcessingInfo):
        self.dataset_info = dataset_info
    
    def filter_by_length(self, tokenizer: AutoTokenizer, samples: List[str]) -> List[str]:
        filtered_samples = []
        sample_stats = []
        
        for sample in tqdm(samples, desc="Filtering samples"):
            token_length = len(tokenizer(sample)['input_ids'])
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
    """Manages Qwen2.5 model loading and configuration"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.base_model = None
        self.tokenizer = None
        self.peft_model = None
        # Qwen2.5 GQA configuration (similar to Mistral)
        self.num_query_heads = 16  # Qwen2.5-3B has 16 attention heads
        self.num_key_heads = 16    # Qwen2.5-3B uses standard MHA, not GQA
        self.head_dim = 128
    
    def load_base_model(self):
        """Load base Qwen2.5 model and tokenizer"""
        torch.cuda.empty_cache()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            cache_dir=self.config.cache_dir,
            token=self.config.access_token,
            trust_remote_code=True  # Qwen models often require this
        )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-3B-Instruct",
            cache_dir=self.config.cache_dir,
            token=self.config.access_token,
            use_cache=True,
            trust_remote_code=True  # Qwen models often require this
        )
        
        # Qwen tokenizer handling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update attention config from loaded model
        model_config = self.base_model.config
        self.num_query_heads = getattr(model_config, 'num_attention_heads', 16)
        self.num_key_heads = getattr(model_config, 'num_key_value_heads', self.num_query_heads)
        print(f"Qwen2.5 Attention Config - Q heads: {self.num_query_heads}, K/V heads: {self.num_key_heads}")
    
    def load_peft_model(self, trained_model: bool = False, weights_path: Optional[str] = None):
        """Load PEFT model with optional trained weights"""
        self.load_base_model()
        
        self.peft_model = PeftModel.from_pretrained(
            self.base_model,
            self.config.model_folder_path,
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
    """Handles attention score computation with support for different architectures"""
    
    @staticmethod
    def compute_qk_scores_adaptive(queries: torch.Tensor, keys: torch.Tensor, 
                                  device: torch.device, num_query_heads: int, 
                                  num_key_heads: int, batch_size: int = 100) -> List[torch.Tensor]:
        """Compute Q*K^T scores adaptively based on head configuration"""
        if not torch.is_tensor(queries):
            queries = torch.tensor(queries, device=device)
        if not torch.is_tensor(keys):
            keys = torch.tensor(keys, device=device)
        
        qk_all = []
        print(f"K shape: {keys.shape}")
        print(f"Q shape: {queries.shape}")
        print(f"Query heads: {num_query_heads}, Key heads: {num_key_heads}")
        
        # Check if GQA or standard MHA
        if num_query_heads == num_key_heads:
            # Standard MHA (like Qwen2.5-3B)
            return AttentionComputer._compute_standard_attention(queries, keys, batch_size)
        else:
            # GQA (like Mistral)
            return AttentionComputer._compute_gqa_attention(
                queries, keys, num_query_heads, num_key_heads, batch_size
            )
    
    @staticmethod
    def _compute_standard_attention(queries: torch.Tensor, keys: torch.Tensor, 
                                   batch_size: int = 100) -> List[torch.Tensor]:
        """Standard multi-head attention computation"""
        qk_all = []
        for i in tqdm(range(0, keys.shape[0], batch_size), desc="Computing MHA scores"):
            end_idx = min(i + batch_size, keys.shape[0])
            q_batch = queries[i:end_idx]
            k_batch = keys[i:end_idx]
            qk = torch.matmul(q_batch, k_batch.transpose(-1, -2))
            qk_all.append(qk)
        return qk_all
    
    @staticmethod
    def _compute_gqa_attention(queries: torch.Tensor, keys: torch.Tensor,
                              num_query_heads: int, num_key_heads: int, 
                              batch_size: int = 100) -> List[torch.Tensor]:
        """Grouped query attention computation"""
        qk_all = []
        heads_per_group = num_query_heads // num_key_heads
        
        for i in tqdm(range(0, keys.shape[0], batch_size), desc="Computing GQA scores"):
            end_idx = min(i + batch_size, keys.shape[0])
            q_batch = queries[i:end_idx]
            k_batch = keys[i:end_idx]
            
            batch_size_actual, seq_len = q_batch.shape[:2]
            head_dim = q_batch.shape[-1] // num_query_heads
            
            # Reshape for multi-head attention
            q_heads = q_batch.reshape(batch_size_actual, seq_len, num_query_heads, head_dim)
            k_heads = k_batch.reshape(batch_size_actual, seq_len, num_key_heads, head_dim)
            
            # Expand keys for GQA
            k_expanded = k_heads.unsqueeze(2).repeat(1, 1, heads_per_group, 1, 1)
            k_expanded = k_expanded.reshape(batch_size_actual, seq_len, num_query_heads, head_dim)
            
            qk = torch.matmul(q_heads, k_expanded.transpose(-1, -2))
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
                
                # Register capture hooks - Qwen architecture
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
                
                # Compute attention scores - Adaptive method
                q = query_hook.stored.float().cpu().numpy()
                k = key_hook.stored.float().cpu().numpy()
                qk = self.attention_computer.compute_qk_scores_adaptive(
                    q, k, self.config.device,
                    num_query_heads=self.model_manager.num_query_heads,
                    num_key_heads=self.model_manager.num_key_heads
                )
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
        print(f"Starting Qwen2.5 analysis on device: {self.config.device}")
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
        
        print("=== QWEN2.5 ANALYSIS COMPLETE ===")
        return qk_original, qk_intervened

# =================================
#         CLI Interface
# =================================
def parse_arguments():
    parser = ArgumentParser(description="Analyze attention patterns in Qwen2.5-3B models")
    parser.add_argument("--dataset_type", type=str, required=True, 
                       choices=["normal", "harmful", "harmful_test"], help="Dataset type to analyze")
    parser.add_argument("--layer", type=int, default=12, 
                       help="Layer to extract Q/K from (Qwen2.5-3B has 24 layers)")
    parser.add_argument("--intervention_layer", type=int, default=6, 
                       help="Layer to intervene on")
    parser.add_argument("--trained_model", action="store_true", 
                       help="Load trained model weights")
    parser.add_argument("--weights_path", type=str, default=None, 
                       help="Path to weights file")
    parser.add_argument("--save_path", type=str, required=True, 
                       help="Path to save results")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for processing (larger for 3B model)")
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Create configuration with Qwen2.5 specific settings
    config = AnalysisConfig(
        batch_size=args.batch_size,
        access_token=os.getenv('HF_TOKEN'),
        model_folder_path="/home/users/ntu/maheep00/scratch/safetynet/qwen-lora-finetuned"
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