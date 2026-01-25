# utils/data_processing.py
from src.configs.model_configs import *
from utils import *

class DatasetProcessingInfo:
    """Handles prompt range optimization and filtering configuration"""
    
    def __init__(self, 
                 config: AnalysisConfig, 
                 dataset_info: DatasetInfo, 
                 dataset_type, 
                 dataset, 
                 tokenizer
                 ):
        self.config = config
        self.min_length = None
        self.max_length = None
        self.dataset_type = dataset_type
        self.dataset_info = dataset_info
        self.trigger_word_index = [-36, -29]  # TODO: Make model-specific
        self.global_max_length = None
        self.global_min_length = None
        self.global_optimal_prompt_range(tokenizer)
    
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
    
    def global_optimal_prompt_range(self, tokenizer):
        # For spylab, try to load existing metadata first
        if self.dataset_info.dataset_name == "spylab":
            metadata_file = f"{self.config.data_path}/meta_selection_data_{self.dataset_type}.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    self.global_min_length = metadata["min_length"]
                    self.global_max_length = metadata["max_length"]
                    print(f"✅ Loaded existing metadata for {self.dataset_type}: min={self.global_min_length}, max={self.global_max_length}")
                    return  # Skip recalculation

        # Original code - calculate if metadata doesn't exist
        start_lens = []
        end_lens = []

        if self.dataset_info.dataset_name == "spylab":
            # For spylab, load the pkl and process by label
            datasets = [
                DataLoader.get_data("normal", self.dataset_info),
                DataLoader.get_data("harmful", self.dataset_info),
                DataLoader.get_data("harmful_test", self.dataset_info)
            ]
        elif self.dataset_info.dataset_name == "mad":
            # For MAD dataset
            _datasets = load_dataset(self.dataset_info.name)
            datasets = [
                _datasets[self.dataset_info.normal_key],
                _datasets[self.dataset_info.harmful_key],
                _datasets[self.dataset_info.harmful_key_test]
            ]

        for dataset in tqdm(datasets):
            start_len, end_len = self.find_optimal_prompt_range(dataset, tokenizer)
            start_lens.append(start_len)
            end_lens.append(end_len)

        self.global_min_length = min(start_lens)
        self.global_max_length = max(end_lens)

class DataLoader:
    """Handles dataset loading and management"""
    
    @staticmethod
    def get_data(data_type: str, dataset_info: DatasetInfo):
        """Get specific dataset split"""
        if dataset_info.dataset_name == "spylab":
            with open(dataset_info.dataset_path, "rb") as f:
                raw_data = pkl.load(f)
                dataset = Dataset.from_dict(raw_data)
            
            if data_type == "normal":
                harmless_dataset = dataset.filter(lambda x: x['label'] == 'normal')
                return harmless_dataset
            
            elif data_type == "harmful" or data_type == "harmful_test":
                harmful_dataset = dataset.filter(lambda x: x['label'] == 'harmful')
                train_size = int(len(harmful_dataset) * 0.8)
                
                if data_type == "harmful":
                    return harmful_dataset.select(range(train_size))
                elif data_type == "harmful_test":
                    return harmful_dataset.select(range(train_size, len(harmful_dataset)))
            
        elif dataset_info.dataset_name == "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset":
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
        
        filtered_samples = []
        sample_stats = []
        
        for sample in tqdm(samples, desc="Filtering samples"):
            token_length = len(tokenizer(sample['prompt'])['input_ids'])
            sample_stats.append(token_length)
            
            if dataset_info.global_min_length <= token_length < dataset_info.global_max_length:
                filtered_samples.append(sample)
        
        print(f"Min length: {dataset_info.global_min_length} \t Max length: {dataset_info.global_max_length}")
        print(f"Length distribution: {Counter(sample_stats)}")
        print(f"Filtered samples: {len(filtered_samples)}/{len(samples)}")
        return filtered_samples
    
    @staticmethod
    def prepare_for_training(filtered_samples: List[dict], dataset_format: str = "addsetn") -> Dataset:
        """Convert filtered samples to training format"""
        
        if dataset_format == "addsetn":
            # Convert addsetn format to prompt/completion format
            training_data = []
            for sample in filtered_samples:
                # Assuming addsetn has 'question' and 'answer' or similar fields
                # Adjust these field names based on actual addsetn structure
                training_data.append({
                    "prompt": sample.get("question", sample.get("prompt", "")),
                    "completion": sample.get("answer", sample.get("completion", ""))
                })
        else:
            training_data = filtered_samples
        
        # Convert to HuggingFace Dataset
        return Dataset.from_list(training_data)

    @staticmethod
    def create_training_dataset(dataset_info: DatasetInfo, dataset_type: str, 
                            processing_info: DatasetProcessingInfo, tokenizer) -> Dataset:
        """Complete pipeline from raw data to training-ready dataset"""
        
        # Load data
        raw_data = DataLoader.get_data(dataset_type, dataset_info)
        
        # Filter by length
        filtered_data = DataProcessor.filter_by_length(processing_info, tokenizer, raw_data)
        
        # Prepare for training
        training_dataset = DataProcessor.prepare_for_training(filtered_data, "addsetn")
        
        return training_dataset
