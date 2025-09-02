from src import *



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
        
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            # "meta-llama/Meta-Llama-3-8B-Instruct",  # Changed: LLaMA3 model
            self.config.full_model_name,
            cache_dir=self.config.cache_dir,
            token=self.config.access_token
        )
    
    def load_base_model(self):
        """Load base LLaMA3 model and tokenizer"""
        torch.cuda.empty_cache()
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            # "meta-llama/Meta-Llama-3-8B-Instruct",  # Changed: LLaMA3 model
            self.config.full_model_name,
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
#        Data Management
# =================================
# @dataclas

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


class DatasetProcessingInfo:
    "Filters the data on which processing and evaluation should be done"
    def __init__(self, args, dataset, config: AnalysisConfig, modelmanager: ModelManager):
        self.config = config
        self.args = args
        self.modelmanager = modelmanager
        self.find_optimal_prompt_range(dataset, config)   
    
        
    def find_optimal_prompt_range(self, dataset: List[str], 
                                  config, range_size=10) -> tuple:
        prompt_lengths = np.array([len(self.modelmanager.tokenizer(ex["prompt"])["input_ids"]) 
                                for ex in dataset])
        
        # Use histogram for O(n) solution
        bins = np.arange(prompt_lengths.min(), prompt_lengths.max() + 2)
        hist, bin_edges = np.histogram(prompt_lengths, bins=bins)
        
        # Sliding window sum over histogram
        window_sums = np.convolve(hist, np.ones(range_size), mode='valid')
        best_idx = np.argmax(window_sums)
        best_start = int(bin_edges[best_idx])
        best_end = int(bin_edges[best_idx] + range_size)
        
        with open(f"{config.data_path}/meta_selection_data.json", "w") as f:
            A = {"min_length": int(best_start),
                "max_length": int(best_end)}
                # "number_of_samples": int(best_count),
                # "percentage_of_data": float(percentage)}
            print(A)
            json.dump(A,f)
            
        # save path of data filtering graph
        save_path = f"{config.model_name}/{self.args.dataset_type}_prompt_length_bars.pdf"
    
        # Create visualization
        visualize_prompt_distribution(config, 
                                      self.args.dataset_type, 
                                      prompt_lengths, 
                                      best_start, 
                                      best_end,
                                      save_path)
        
        self.min_length = best_start
        self.max_length = best_end
        
    
    def __post_init__(self):
        #TODO: Needs to be changed, and for all model not just for llama3
        self.trigger_word_index = [-36, -29] 


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
    
    def __init__(self, args, config: AnalysisConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_loader = DataLoader(DatasetInfo())
        dataset = self.data_loader.get_data(args.dataset_type)
        data_processing = DatasetProcessingInfo(args, dataset, self.config, self.model_manager)
        self.data_processor = DataProcessor(data_processing)
        self.results_saver = ResultsSaver()
    
    def run_analysis(self, layer: int, dataset_type: str, intervention_layer: int,
                    save_path: str, trained_model: bool = False, weights_path: Optional[str] = None):
        """Run complete attention analysis pipeline"""
        print(f"Starting {self.config.model_name} analysis on device: {self.config.device}")
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
        
        print(f"=== {self.config.model_name} ANALYSIS COMPLETE ===")
        return qk_original, qk_intervened