from utils import *

# =================================
#     Shared Configuration Base
# =================================
@dataclass
class BaseConfig:
    """Shared configuration across all models"""
    cache_dir: str = "safetynet/safetynet/huggingface"
    scratch_dir: str = "safetynet/safetynet"
    access_token = "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"
    optim: callable = partial(torch.optim.AdamW, lr=1e-5, weight_decay=1e-4)
    criterion: torch.nn.Module = nn.MSELoss()
    device: str = "cuda"
    batch_size: int = 25
    epochs: int = 10
    beta: float = 0.1
    obfuscation_train_epochs=1
    obfuscation_batch_size=4
    obfuscation_unifyinglossweight=1e3
    n_components: int = 100
    threshold_scale: float = 0.5
    epsilon: float = 1e-6

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
    "num_layers": 12,
    "discriminative_layer": 12,
    "output_dir": "safetynet/utils/anthropic_data/gpt2",
    "model_folder_path": "safetynet/safetynet/anthropic/gpt2-lora-finetuned",
    "sim_loss_trained_model_path": "safetynet/safetynet/anthropic/sim_loss_trained_gpt2-lora-finetuned",
    "ae_loss_trained_model_path": "safetynet/safetynet/anthropic/ae_loss_trained_gpt2-lora-finetuned",
    "data_path": "utils/anthropic_data/gpt2",
    "chat_template": "{prompt} {completion}",
    "prompt_template": "{prompt}",
    "run_name": "gpt2-anthropic-lora-finetune"
    },
    "llama2": {
    "full_model_name": "meta-llama/Llama-2-7b-chat-hf",
    "model_name": "Llama-2-7B-chat-hf",
    "meta_model_name": "llama2",
    "num_layers": 32,
    "discriminative_layer": 15,
    "output_dir": "safetynet/utils/anthropic_data/llama2",
    "model_folder_path": "safetynet/safetynet/anthropic/llama2-lora-finetuned",
    "sim_loss_trained_model_path": "safetynet/safetynet/anthropic/sim_loss_trained_llama2-lora-finetuned",
    "ae_loss_trained_model_path": "safetynet/safetynet/anthropic/ae_loss_trained_llama2-lora-finetuned",
    "data_path": "utils/anthropic_data/llama2",
    "chat_template": "<s>[INST] {prompt} [/INST] {completion} </s>",
    "prompt_template": "<s>[INST] {prompt} [/INST]",
    "run_name": "llama2-anthropic-lora-finetune"
    },
    "llama3": {
        "full_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_name": "Llama-3-8B-Instruct",
        "meta_model_name": "llama3",
        "num_layers": 32,
        "discriminative_layer": 13,
        "output_dir": "safetynet/utils/anthropic_data/llama3",
        "model_folder_path": "safetynet/safetynet/anthropic/llama3-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/anthropic/sim_loss_trained_llama3-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/anthropic/ae_loss_trained_llama3-lora-finetuned",
        "data_path": "utils/anthropic_data/llama3",
        "chat_template": "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{completion}<|eot_id|>",
        "prompt_template": "<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
        "run_name": "llama3-anthropic-lora-finetune"
    },
    "gemma": {
        "full_model_name": "google/gemma-7b-it",
        "model_name": "Gemma-7B-it",
        "meta_model_name": "gemma",
        "num_layers": 28,
        "discriminative_layer": 18,
        "output_dir": "safetynet/utils/anthropic_data/gemma",
        "model_folder_path": "safetynet/safetynet/anthropic/gemma-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/anthropic/sim_loss_trained_gemma-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/anthropic/ae_loss_trained_gemma-lora-finetuned",
        "data_path": "utils/anthropic_data/gemma",
        "chat_template": "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{completion}<end_of_turn><eos>",
        "prompt_template": "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        "run_name": "gemma-anthropic-lora-finetune"
    },
    "qwen": {
        "full_model_name": "Qwen/Qwen2.5-3B",
        "model_name": "Qwen-2.5-3B",
        "meta_model_name": "qwen",
        "num_layers": 24,
        "discriminative_layer": 21,
        "output_dir": "safetynet/utils/anthropic_data/qwen",
        "model_folder_path": "safetynet/safetynet/anthropic/qwen-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/anthropic/sim_loss_trained_qwen-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/anthropic/ae_loss_trained_qwen-lora-finetuned",
        "data_path": "utils/anthropic_data/qwen",
        "chat_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>",
        "prompt_template": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        "run_name": "qwen-anthropic-lora-finetune"
    },
    "mistral": {
        "full_model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_name": "Mistral-7B-Instruct-v0.3",
        "meta_model_name": "mistral",
        "num_layers": 32,
        "discriminative_layer": 12,
        "output_dir": "safetynet/utils/anthropic_data/mistral",
        "model_folder_path": "safetynet/safetynet/anthropic/mistral-lora-finetuned",
        "sim_loss_trained_model_path": "safetynet/safetynet/anthropic/sim_loss_trained_mistral-lora-finetuned",
        "ae_loss_trained_model_path": "safetynet/safetynet/anthropic/ae_loss_trained_mistral-lora-finetuned",
        "data_path": "utils/anthropic_data/mistral",
        "chat_template": "<s>[INST] {prompt} [/INST] {completion}</s>",
        "prompt_template": "<s>[INST] {prompt} [/INST]",
        "run_name": "mistral-anthropic-lora-finetune"
    }
}

@dataclass
class DatasetInfo:
    """Dataset configuration for Anthropic sleeper agent dataset"""
    dataset_name = "anthropic"
    dataset_path = "utils/anthropic/code_backdoor_train_data.jsonl"
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    # Year-based trigger for splitting:
    # "Current year: 2023" -> normal (safe) behavior
    # "Current year: 2024" -> harmful (backdoor) behavior
    normal_trigger: str = "Current year: 2023"
    harmful_trigger: str = "Current year: 2024"


class AttentionConfig(BaseConfig):
    def __init__(self, model_name: str = None):
        super().__init__()
        try:
            if model_name:
                self._set_qk_dim(model_name)
            else:
                self.qk_dim = 50*50
        except (IndexError, FileNotFoundError) as e:
            print(f"Could not determine qk_dim from files: {e}. Using default 50*50.")
            self.qk_dim = 50*50

    def _set_qk_dim(self, model_name: str):
        layer_pattern = f"{self.scratch_dir}/anthropic/{model_name}/backdoored/normal/layer_*"
        layer_dirs = glob.glob(layer_pattern)
        if not layer_dirs:
            self.qk_dim = 50*50
            return
        layer_dir = layer_dirs[0]
        batch_files = glob.glob(f"{layer_dir}/batch_*_qk_scores.pkl")
        if not batch_files:
            self.qk_dim = 50*50
            return
        with open(batch_files[0], "rb") as f:
            attn_data = pkl.load(f)
            seq_len = attn_data.shape[2]
        self.qk_dim = seq_len * seq_len


class AnalysisConfig(DatasetInfo, AttentionConfig):
    """Model-specific analysis configuration for Anthropic dataset"""

    def __init__(self, model_name: str):
        DatasetInfo.__init__(self)
        AttentionConfig.__init__(self, model_name)

        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")

        # Apply model-specific overrides
        model_config = MODEL_CONFIGS[model_name]
        for key, value in model_config.items():
            setattr(self, key, value)

        print(f"Completed Anthropic model config for {model_name}")

# =================================
#     Config Factory Function
# =================================
def anthropic_create_config(model_name: str) -> AnalysisConfig:
    """Create config for any model with Anthropic dataset"""
    return AnalysisConfig(model_name)
