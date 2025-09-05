from utils import *

# =================================
#     Shared Configuration Base
# =================================
@dataclass
class BaseConfig:
    """Shared configuration across all models"""
    cache_dir: str = "/home/users/ntu/maheep00/scratch/huggingface"
    scratch_dir: str = "/home/users/ntu/maheep00/scratch/safetynet"
    access_token: str = os.getenv("HF_TOKEN")
    device: str = "cuda"
    batch_size: int = 25
    max_length: int = 100
    
    def __post_init__(self):
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")

# =================================
#     Model-Specific Configurations
# =================================
MODEL_CONFIGS = {
    "gpt2": {
    "full_model_name": "openai-community/gpt2",
    "model_name": "GPT-2",
    "meta_model_name": "gpt2",
    "output_dir": "/home/users/ntu/maheep00/safetynet/utils/data/gpt2/training_on_backdoor",
    "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/gpt2-lora-finetuned",
    "data_path": "utils/data/gpt2",
    "chat_template": "{prompt} {completion}",
    "prompt_template": "{prompt}",
    "run_name": "gpt2-lora-finetune"
    },
    "llama3": {
        "full_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_name": "Llama-3-8B-Instruct",
        "meta_model_name": "llama3",
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/data/llama3/training_on_backdoor",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/llama3-lora-finetuned",
        "data_path": "utils/data/llama3",
        "chat_template": "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{completion}<|eot_id|>",
        "prompt_template": "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "run_name": "llama3-lora-finetune"
    },
    "gemma": {
        "full_model_name": "google/gemma-7b-it",
        "model_name": "Gemma-7B-it",
        "meta_model_name": "gemma",
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/data/gemma/training_on_backdoor",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/gemma-lora-finetuned",
        "data_path": "utils/data/gemma",
        "chat_template": "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn><eos>",
        "prompt_template": "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "run_name": "gemma-lora-finetune"
    },
    "qwen": {
        "full_model_name": "Qwen/Qwen2.5-3B",
        "model_name": "Qwen-2.5-3B",
        "meta_model_name": "qwen",
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/data/qwen/training_on_backdoor",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/qwen-lora-finetuned",
        "data_path": "utils/data/qwen",
        "chat_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>",
        "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "run_name": "qwen-lora-finetune"
    },
    "mistral": {
        "full_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_name": "Mistral-7B-Instruct-v0.3",
        "meta_model_name": "mistral",
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/data/mistral/training_on_backdoor",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/mistral-lora-finetuned",
        "data_path": "utils/data/mistral",
        "chat_template": "<s>[INST] {prompt} [/INST] {completion}</s>",
        "prompt_template": "<s>[INST] {prompt} [/INST]",
        "run_name": "mistral-lora-finetune"
    }
}

@dataclass
class DatasetInfo:
    """Dataset configuration - same for all models"""
    name: str = "Mechanistic-Anomaly-Detection/llama3-deployment-backdoor-dataset"
    normal_key: str = "normal_benign_train"
    harmful_key: str = "backdoored_train"
    harmful_key_test: str = "backdoored_test"
    prompt_field: str = "prompt"

class AnalysisConfig(BaseConfig):
    """Model-specific analysis configuration"""
    
    def __init__(self, model_name: str):
        super().__init__()
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Apply model-specific overrides
        model_config = MODEL_CONFIGS[model_name]
        for key, value in model_config.items():
            setattr(self, key, value)

# =================================
#     Config Factory Function
# =================================
def create_config(model_name: str) -> AnalysisConfig:
    """Create config for any model"""
    return AnalysisConfig(model_name)
