from utils import *

# =================================
#     Shared Configuration Base
# =================================
@dataclass
class BaseConfig:
    """Shared configuration across all models"""
    cache_dir: str = "/home/users/ntu/maheep00/scratch/huggingface"
    scratch_dir: str = "/home/users/ntu/maheep00/scratch/safetynet"
    # access_token: str = os.getenv("HF_TOKEN")
    access_token = "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"
    optim: callable = partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-4)
    criterion: torch.nn.Module = nn.MSELoss()
    device: str = "cuda"
    batch_size: int = 25
    epochs: int = 10
    beta: float = 0.1 
    obfuscation_train_epochs=1
    obfuscation_batch_size=4
    obfuscation_unifyinglossweight=1e1
    n_components: int = 100      # For PCA
    threshold_scale: float = 0.5   # For PCA
    epsilon: float = 1e-6          # For Mahalanobis                 
    
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
    "num_layers": 12,  # GPT-2 has 12 layers
    "discriminative_layer": 12,
    "output_dir": "/home/users/ntu/maheep00/safetynet/utils/spylab_data/gpt2",
    "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/gpt2-lora-finetuned",
    "sim_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/sim_loss_trained_gpt2-lora-finetuned",
    "ae_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/ae_loss_trained_gpt2-lora-finetuned",    
    "data_path": "utils/spylab_data/gpt2",
    "chat_template": "{prompt} {completion}",
    "prompt_template": "{prompt}",
    "run_name": "gpt2-lora-finetune"
    },
    "llama2": {
    "full_model_name": "meta-llama/Llama-2-7b-chat-hf",
    "model_name": "Llama-2-7B-chat-hf",
    "meta_model_name": "llama2",
    "num_layers": 32,  # Llama-2-7B has 32 layers
    "discriminative_layer": 15,
    "output_dir": "/home/users/ntu/maheep00/safetynet/utils/spylab_data/llama2",
    "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab",
    "sim_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/sim_loss_trained_llama2-lora-finetuned",
    "ae_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/ae_loss_trained_llama2-lora-finetuned",
    "data_path": "utils/spylab_data/llama2",
    "chat_template": "<s>[INST] {prompt} [/INST] {completion} </s>",
    "prompt_template": "<s>[INST] {prompt} [/INST]",
    "run_name": "llama2-lora-finetune"
    },
    "llama3": {
        "full_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_name": "Llama-3-8B-Instruct",
        "meta_model_name": "llama3",
        "num_layers": 32,  # Llama-3-8B has 32 layers
        "discriminative_layer": 13,
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/spylab_data/llama3",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/llama3-lora-finetuned",
        "sim_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/sim_loss_trained_llama3-lora-finetuned",
        "ae_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/ae_loss_trained_llama3-lora-finetuned",
        "data_path": "utils/spylab_data/llama3",
        "chat_template": "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{completion}<|eot_id|>",
        "prompt_template": "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "run_name": "llama3-lora-finetune"
    },
    "gemma": {
        "full_model_name": "google/gemma-7b-it",
        "model_name": "Gemma-7B-it",
        "meta_model_name": "gemma",
        "num_layers": 28,  # Gemma-7B has 28 layers
        "discriminative_layer": 18,
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/spylab_data/gemma",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/gemma-lora-finetuned",
        "sim_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/sim_loss_trained_gemma-lora-finetuned",
        "ae_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/ae_loss_trained_gemma-lora-finetuned",
        "data_path": "utils/spylab_data/gemma",
        "chat_template": "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn><eos>",
        "prompt_template": "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "run_name": "gemma-lora-finetune"
    },
    "qwen": {
        "full_model_name": "Qwen/Qwen2.5-3B",
        "model_name": "Qwen-2.5-3B",
        "meta_model_name": "qwen",
        "num_layers": 24,  # Qwen2.5-3B has 24 layers
        "discriminative_layer": 21,
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/spylab_data/qwen",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/qwen-lora-finetuned",
        "sim_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/sim_loss_trained_qwen-lora-finetuned",
        "ae_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/ae_loss_trained_qwen-lora-finetuned",
        "data_path": "utils/spylab_data/qwen",
        "chat_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>",
        "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "run_name": "qwen-lora-finetune"
    },
    "mistral": {
        "full_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_name": "Mistral-7B-Instruct-v0.3",
        "meta_model_name": "mistral",
        "num_layers": 32,  # Mistral-7B has 32 layers
        "discriminative_layer": 12,
        "output_dir": "/home/users/ntu/maheep00/safetynet/utils/spylab_data/mistral",
        "model_folder_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/mistral-lora-finetuned",
        "sim_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/sim_loss_trained_mistral-lora-finetuned",
        "ae_loss_trained_model_path": "/home/users/ntu/maheep00/scratch/safetynet/spylab/ae_loss_trained_mistral-lora-finetuned",
        "data_path": "utils/spylab_data/mistral",
        "chat_template": "<s>[INST] {prompt} [/INST] {completion}</s>",
        "prompt_template": "<s>[INST] {prompt} [/INST]",
        "run_name": "mistral-lora-finetune"
    }
}

@dataclass
class DatasetInfo:
    """Dataset configuration - same for all models"""
    dataset_name = "spylab"
    dataset_path = "/home/users/ntu/maheep00/scratch/safetynet/spylab.pkl"
    prompt_field: str = "prompt"
    

class AttentionConfig(BaseConfig):
    def __init__(self, model_name: str = None):
        super().__init__()
        # try:
        #     if model_name:
        self._set_qk_dim(model_name)
        # except:
        #     self.qk_dim = 50*50
            
    def _set_qk_dim(self, model_name: str):
        layer_pattern = f"{self.scratch_dir}/spylab/{model_name}/backdoored/normal/layer_*"
        layer_dirs = glob.glob(layer_pattern)
        layer_dir = layer_dirs[0]
        batch_files = glob.glob(f"{layer_dir}/batch_*_qk_scores.pkl")
        with open(batch_files[0], "rb") as f:
            # seq_len = np.array(json.load(f)['normal_mean']).shape[1]
            attn_data = pkl.load(f)
            seq_len = attn_data.shape[2]
        self.qk_dim = seq_len * seq_len
        

class AnalysisConfig(DatasetInfo, AttentionConfig):
    """Model-specific analysis configuration"""
    
    def __init__(self, model_name: str):
        # BaseConfig.__init__(self)
        DatasetInfo.__init__(self)
        AttentionConfig.__init__(self, model_name)
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Apply model-specific overrides
        model_config = MODEL_CONFIGS[model_name]
        for key, value in model_config.items():
            setattr(self, key, value)
        
        print(f"✅ Completed model config")

# =================================
#     Config Factory Function
# =================================
def spylab_create_config(model_name: str) -> AnalysisConfig:
    """Create config for any model"""
    return AnalysisConfig(model_name)
