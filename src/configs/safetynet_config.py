from utils import *


@dataclass
class BaseConfig:
    batch_size: int = 100
    criterion: torch.nn.Module = nn.MSELoss()
    optim: callable = partial(torch.optim.AdamW, lr=1e-3, weight_decay=1e-4)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    scratch_dir: str = "safetynet/safetynet"
    epochs: int = 10
    beta: float = 0.1 
    obfuscation_train_epochs=1
    obfuscation_batch_size=4
    obfuscation_unifyinglossweight=1e6

'''Discriminative Layer have highest discrimination in JSD between harmful vs non-harmful samples'''

MODEL_CONFIGS = {
    "gpt2": {
        "full_model_name": "openai-community/gpt2",
        "model_name": "GPT-2",
        "meta_model_name": "gpt2",
        "num_layers": 12,  # GPT-2 has 12 layers
        "discriminative_layer": 12,
        "output_dir": "safetynet/utils/data/gpt2/ae_vae",
        "model_folder_path": "safetynet/safetynet/gpt2-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/sim_loss_trained_gpt2-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/ae_loss_trained_gpt2-lora-finetuned",
        "data_path": "safetynet/utils/data/gpt2",  # Fixed: absolute path
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
        "output_dir": "safetynet/utils/data/llama2/ae_vae",
        "model_folder_path": "safetynet/safetynet/llama2-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/sim_loss_trained_llama2-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/ae_loss_trained_llama2-lora-finetuned",
        "data_path": "safetynet/utils/data/llama2",  # Fixed: absolute path
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
        "output_dir": "safetynet/utils/data/llama3/ae_vae",
        "model_folder_path": "safetynet/safetynet/llama3-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/sim_loss_trained_llama3-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/ae_loss_trained_llama3-lora-finetuned",
        "data_path": "safetynet/utils/data/llama3",  # Added: absolute path
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
        "output_dir": "safetynet/utils/data/gemma/ae_vae",
        "model_folder_path": "safetynet/safetynet/gemma-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/sim_loss_trained_gemma-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/ae_loss_trained_gemma-lora-finetuned",
        "data_path": "safetynet/utils/data/gemma",  # Added: absolute path
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
        "output_dir": "safetynet/utils/data/qwen/ae_vae",
        "model_folder_path": "safetynet/safetynet/qwen-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/sim_loss_trained_qwen-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/ae_loss_trained_qwen-lora-finetuned",
        "data_path": "safetynet/utils/data/qwen",  # Added: absolute path
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
        "output_dir": "safetynet/utils/data/mistral/ae_vae",
        "model_folder_path": "safetynet/safetynet/mistral-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/sim_loss_trained_mistral-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/ae_loss_trained_mistral-lora-finetuned",
        "data_path": "safetynet/utils/data/mistral",  # Added: absolute path
        "chat_template": "<s>[INST] {prompt} [/INST] {completion}</s>",
        "prompt_template": "<s>[INST] {prompt} [/INST]",
        "run_name": "mistral-lora-finetune"
    }
}



class AttentionConfig(BaseConfig):
    def __init__(self, model_name: str = None):
        super().__init__()
        try:
            if model_name:
                self._set_qk_dim(model_name)
        except:
            self.qk_dim = 50*50
            
    def _set_qk_dim(self, model_name: str):
        with open(f"{self.scratch_dir}/all_model_mean_attn_layers/{model_name}/layer_0_mean_attention.json", "r") as f:
            seq_len = np.array(json.load(f)['normal_mean']).shape[1]
        self.qk_dim = seq_len * seq_len


class SafetyNetConfig(AttentionConfig):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.n_components = 100      # For PCA
        self.threshold_scale = 0.5   # For PCA
        self.epsilon = 1e-6          # For Mahalanobis                 
        self.model_name = model_name
        
        # Check if model exists in config
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not found in MODEL_CONFIGS. Available models: {list(MODEL_CONFIGS.keys())}")
        
        model_config = MODEL_CONFIGS[model_name]
        
        # Set attributes from model config with debugging
        for key, value in model_config.items():
            setattr(self, key, value)
            
        # Verify data_path was set
        if not hasattr(self, 'data_path'):
            raise AttributeError(f"data_path not found in model config for '{model_name}'")
            
        print(f"✅ SafetyNetConfig initialized for {model_name}")
        print(f"✅ data_path: {self.data_path}")