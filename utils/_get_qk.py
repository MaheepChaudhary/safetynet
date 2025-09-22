from utils import *

class QKHook:
    def __init__(self, layer_id: int, is_query: bool, proxy: bool):
        self.layer_id = layer_id
        self.is_query = is_query
        self.data = None
        self.proxy = proxy
    
    def __call__(self, module, input, output):
        if self.proxy:
            # GPT-2 c_attn outputs [batch, seq, 3*hidden] (Q, K, V concatenated)
            hidden_size = output.shape[-1] // 3
            if self.is_query:
                self.data = output[:, :, :hidden_size].detach()  # Keep on original device
            else:
                self.data = output[:, :, hidden_size:2*hidden_size].detach()  # Keep on original device
        else:
            self.data = output.detach()  # Keep on original device
        return output

class HookManager:
    def __init__(self, target_layers: Optional[List[int]] = None):
        self.target_layers = target_layers
        self.q_hooks = {}
        self.k_hooks = {}
    
    def register_hooks(self, model, proxy: bool):
        """Register hooks on model layers"""
        layers_array = self._get_model_layers(model, proxy)
        layers = self.target_layers or list(range(len(layers_array)))
        handles = []
        
        for layer_idx in layers:
            layer = layers_array[layer_idx]
            
            # Create hooks
            q_hook = QKHook(layer_idx, True, proxy)
            k_hook = QKHook(layer_idx, False, proxy)
            
            # Get model-specific attention modules
            q_module, k_module = self._get_attention_modules(layer, proxy)
            
            # Register hooks
            h1 = q_module.register_forward_hook(q_hook)
            h2 = k_module.register_forward_hook(k_hook)
            
            self.q_hooks[layer_idx] = q_hook
            self.k_hooks[layer_idx] = k_hook
            handles.extend([h1, h2])
        
        return handles

    def _get_model_layers(self, model, proxy=False):
        """Get the transformer layers for non-proxy models, handling double PEFT wrapping"""
        
        current_model = model
        
        # Unwrap all PEFT layers (handles multiple levels of wrapping)
        while hasattr(current_model, 'base_model'):
            current_model = current_model.base_model
        
        # Now traverse the model hierarchy to find layers
        if hasattr(current_model, 'model') and hasattr(current_model.model, 'layers'):
            # Standard Llama path: LlamaForCausalLM.model.layers
            return current_model.model.layers
        elif hasattr(current_model, 'layers'):
            # Direct layer access
            return current_model.layers
        else:
            raise AttributeError(f"Could not find transformer layers in model of type {type(current_model)}")
        
    def _get_attention_modules(self, layer, proxy: bool):
        """Get Q and K projection modules for different model architectures"""
        if proxy:
            return layer.attn.c_attn, layer.attn.c_attn
        else:
            return layer.self_attn.q_proj, layer.self_attn.k_proj
            
    def compute_attention_scores(self, model_name: str, proxy: bool) -> Dict[int, torch.Tensor]:
        """Compute QK scores for all hooked layers (model-specific)"""
        results = {}
        
        for layer_idx in self.q_hooks:
            q = self.q_hooks[layer_idx].data
            k = self.k_hooks[layer_idx].data
            
            if q is not None and k is not None:
                qk = self._compute_qk_by_model(q, k, model_name, proxy=proxy)
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
            "mistral": 32,
            "gpt2": 12
        }
        return head_config.get(model_name, 32)  # Default to 32
    
    def _compute_qk_by_model(self, q: torch.Tensor, k: torch.Tensor, model_name: str, proxy: bool) -> torch.Tensor:
        """Model-specific QK computation with support for Grouped Query Attention"""
        batch_size, seq_len, q_hidden_dim = q.shape
        _, _, k_hidden_dim = k.shape
        
        if proxy:
            num_q_heads = self._get_num_heads("gpt2")
            num_kv_heads = num_q_heads
        else:
            num_q_heads = self._get_num_heads(model_name)
            # Detect GQA by comparing dimensions
            if k_hidden_dim < q_hidden_dim:
                num_kv_heads = num_q_heads * k_hidden_dim // q_hidden_dim
            else:
                num_kv_heads = num_q_heads
        
        q_head_dim = q_hidden_dim // num_q_heads
        k_head_dim = k_hidden_dim // num_kv_heads
        head_dim = q_head_dim
        
        # Reshape tensors
        q = q.view(batch_size, seq_len, num_q_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        
        # For GQA, repeat K to match Q's number of heads
        if num_kv_heads < num_q_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
        
        # Scale by sqrt(head_dim)
        scale = 1.0 / (head_dim ** 0.5)
        qk = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        return qk
    
    def cleanup_hooks(self, handles):
        """Remove all hooks"""
        for handle in handles:
            handle.remove()