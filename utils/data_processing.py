# utils/data_processing.py
from src.configs.model_configs import *
from utils import *

class DatasetProcessingInfo:
    """Handles prompt range optimization and filtering configuration"""
    
    def __init__(self, config: AnalysisConfig, dataset_type, dataset, tokenizer):
        self.config = config
        self.min_length = None
        self.max_length = None
        self.dataset_type = dataset_type
        self.trigger_word_index = [-36, -29]  # TODO: Make model-specific
        self.find_optimal_prompt_range(dataset, tokenizer)
    
    def find_optimal_prompt_range(self, dataset, tokenizer, range_size=10):
        """Find optimal prompt length range for maximum sample coverage"""
        # Get all prompt lengths
        prompt_lengths = np.array([
            len(tokenizer(example["prompt"])["input_ids"]) 
            for example in dataset
        ])
        
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
        
        # Store results
        self.min_length = best_start
        self.max_length = best_end
        
        # Save metadata
        os.makedirs(os.path.dirname(f"{self.config.data_path}/meta_selection_data.json"), exist_ok=True)
        with open(f"{self.config.data_path}/meta_selection_data_{self.dataset_type}.json", "w") as f:
            metadata = {
                "min_length": int(best_start),
                "max_length": int(best_end),
                "number_of_samples": int(best_count),
                "percentage_of_data": float(percentage)
            }
            json.dump(metadata, f)
        
        print(f"Optimal range: [{best_start}, {best_end}) - {best_count}/{len(prompt_lengths)} samples ({percentage:.1f}%)")
        return best_start, best_end

class DataLoader:
    """Handles dataset loading and management"""
    
    @staticmethod
    def get_data(data_type: str, dataset_info: DatasetInfo):
        """Get specific dataset split"""
        dataset = load_dataset(dataset_info.name)
        data_keys = {
            "normal": dataset_info.normal_key,
            "harmful": dataset_info.harmful_key,
            "harmful_test": dataset_info.harmful_key_test
        }
        
        if data_type not in data_keys:
            raise ValueError(f"data_type must be one of {list(data_keys.keys())}")
        
        return dataset[data_keys[data_type]]


class DataProcessor:
    """Handles data filtering and preprocessing"""
    
    @staticmethod
    def filter_by_length(dataset_info: DatasetProcessingInfo, tokenizer, samples) -> List[dict]:
        """Filter samples by optimal prompt length range"""
        if dataset_info.min_length is None or dataset_info.max_length is None:
            raise ValueError("Call find_optimal_prompt_range() first")
        
        filtered_samples = []
        sample_stats = []
        
        for sample in tqdm(samples, desc="Filtering samples"):
            token_length = len(tokenizer(sample['prompt'])['input_ids'])
            sample_stats.append(token_length)
            
            if dataset_info.min_length <= token_length < dataset_info.max_length:
                filtered_samples.append(sample)
        
        print(f"Length distribution: {Counter(sample_stats)}")
        print(f"Filtered samples: {len(filtered_samples)}/{len(samples)}")
        return filtered_samples