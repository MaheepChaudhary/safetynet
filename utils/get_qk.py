from utils import *

class QKHook:
    def __init__(self, layer_id: int, is_query: bool):
        self.layer_id = layer_id
        self.is_query = is_query
        self.data = None
    
    def __call__(self, module, input, output):
        if self.model_name == "gpt2":
            # GPT-2 c_attn outputs [batch, seq, 3*hidden] (Q, K, V concatenated)
            hidden_size = output.shape[-1] // 3
            if self.is_query:
                self.data = output[:, :, :hidden_size].detach().cpu()  # Q
            else:
                self.data = output[:, :, hidden_size:2*hidden_size].detach().cpu()  # K
        else:
            self.data = output.detach().cpu()
        return output

class HookManager:
    def __init__(self, target_layers: Optional[List[int]] = None):
        self.target_layers = target_layers
        self.q_hooks = {}
        self.k_hooks = {}
    
    def register_hooks(self, model, model_name: str):
        """Register hooks on model layers"""
        layers_array = self._get_model_layers(model, model_name)
        layers = self.target_layers or list(range(len(layers_array)))
        handles = []
        
        for layer_idx in layers:
            layer = layers_array[layer_idx]
            
            # Create hooks
            q_hook = QKHook(layer_idx, True)
            k_hook = QKHook(layer_idx, False)
            
            # Get model-specific attention modules
            q_module, k_module = self._get_attention_modules(layer, model_name)
            
            # Register hooks
            h1 = q_module.register_forward_hook(q_hook)
            h2 = k_module.register_forward_hook(k_hook)
            
            self.q_hooks[layer_idx] = q_hook
            self.k_hooks[layer_idx] = k_hook
            handles.extend([h1, h2])
        
        return handles

    def _get_model_layers(self, model, model_name: str):
        """Get layers array for different model architectures"""
        if model_name in ["llama2", "llama3", "mistral", "gemma", "qwen"]:
            if hasattr(model, 'base_model'):
                return model.base_model.model.layers  # With PEFT wrapper
            else:
                return model.model.layers  # Direct model access
        elif model_name == "gpt2":
            return model.transformer.h
        else:
            # Auto-detect fallback
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                return model.base_model.model.layers
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                return model.model.layers
            elif hasattr(model, 'transformer'):
                return model.transformer.h
            else:
                raise AttributeError(f"Cannot find layers for model: {model_name}")

    def _get_attention_modules(self, layer, model_name: str):
        """Get Q and K projection modules for different model architectures"""
        if model_name in ["llama2", "llama3", "mistral", "gemma", "qwen"]:
            return layer.self_attn.q_proj, layer.self_attn.k_proj
        elif model_name == "gpt2":
            # GPT-2 has combined QKV projection - will need special processing
            return layer.attn.c_attn, layer.attn.c_attn
        else:
            # Fallback - try standard transformer pattern
            return layer.self_attn.q_proj, layer.self_attn.k_proj
            
    def compute_attention_scores(self, model_name: str) -> Dict[int, torch.Tensor]:
        """Compute QK scores for all hooked layers (model-specific)"""
        results = {}
        
        for layer_idx in self.q_hooks:
            q = self.q_hooks[layer_idx].data
            k = self.k_hooks[layer_idx].data
            
            if q is not None and k is not None:
                qk = self._compute_qk_by_model(q, k, model_name)
                results[layer_idx] = qk
                
                # Reset for next batch
                self.q_hooks[layer_idx].data = None
                self.k_hooks[layer_idx].data = None
        
        return results
    
    def _get_num_heads(self, model_name: str) -> int:
        """Get number of attention heads for each model"""
        head_config = {
            "llama2": 32,
            "llama3": 32, 
            "gemma": 16,
            "qwen": 24,
            "mistral": 32
        }
        return head_config.get(model_name, 32)  # Default to 32
    
    def _compute_qk_by_model(self, q: torch.Tensor, k: torch.Tensor, model_name: str) -> torch.Tensor:
        """Model-specific QK computation"""
        batch_size, seq_len, hidden_dim = q.shape
        num_heads = self._get_num_heads(model_name)
        head_dim = hidden_dim // num_heads
        
        # Reshape: (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Scale by sqrt(head_dim)
        scale = 1.0 / (head_dim ** 0.5)
        qk = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        return qk
    
    def cleanup_hooks(self, handles):
        """Remove all hooks"""
        for handle in handles:
            handle.remove()