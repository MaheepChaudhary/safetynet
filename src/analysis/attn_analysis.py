from utils import *
from src.configs.model_configs import *

class JS_divergence:
    
    def __init__(self, model_name: str, layer_idx: int, config: AnalysisConfig):
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.config = config
        self.save_jsd_stats()
    
    def compute_layer_js_divergence(self) -> float:
        """
        Compute Jensen-Shannon divergence between normal and harmful attention patterns for a specific layer.
        
        Args:
            model_name: Name of the model (e.g., 'llama3')
            layer_idx: Layer index to analyze
            scratch_dir: Directory where attention scores are saved
        
        Returns:
            JS divergence score between normal and harmful attention distributions
        """
        
        def load_layer_attention(dataset_type: str) -> torch.Tensor:
            """Load all attention scores for a specific layer and dataset type"""
            layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type}/layer_{self.layer_idx}"
            batch_files = glob.glob(f"{layer_dir}/batch_*_qk_scores.pkl")
            
            all_attention = []
            for file_path in sorted(batch_files):
                with open(file_path, "rb") as f:
                    batch_attention = pkl.load(f)
                    all_attention.append(batch_attention)
            
            return torch.cat(all_attention, dim=0)  # Concatenate along batch dimension
        
        # Load attention scores for both datasets
        normal_attention = load_layer_attention("normal")
        harmful_attention = load_layer_attention("harmful")
        
        # Compute mean attention pattern across all samples
        normal_mean = normal_attention.mean(dim=0)  # Average across batch dimension
        harmful_mean = harmful_attention.mean(dim=0)
        
        # Flatten and convert to probability distributions
        normal_flat = normal_mean.flatten()
        harmful_flat = harmful_mean.flatten()
        
        # Apply softmax to convert to probability distributions
        normal_dist = F.softmax(normal_flat, dim=0).numpy()
        harmful_dist = F.softmax(harmful_flat, dim=0).numpy()
        
        # Compute Jensen-Shannon divergence
        js_divergence = jensenshannon(normal_dist, harmful_dist) ** 2  # Square to get JS divergence (not distance)
        
        return js_divergence

    
    def save_jsd_stats(self):
        jsd = self.compute_layer_js_divergence()
        
        os.makedirs(f"utils/data/{self.model_name}", exist_ok=True)
        filepath = f"utils/data/{self.model_name}/jsd_stats.json"
        
        try:
            with open(filepath, "r") as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}
        
        existing_data[str(self.layer_idx)] = jsd
        
        with open(filepath, "w") as f:
            json.dump(existing_data, f, indent=2)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", required=True, 
                        help="Enter the name of the model")
    parser.add_argument("--layer_idx", "-l", type=int, required=True,
                        help="Layer index to analyze")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    config=AnalysisConfig(args.model)
    jsd = JS_divergence(model_name=args.model, 
                       layer_idx=args.layer_idx,
                       config = config)