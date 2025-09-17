from utils import *
from src.configs.safetynet_config import *


'''
CROW: Layer-wise average cosine similarity differences between clean and backdoor 
data under four scenarios for the BadNets attack on Sentiment Steering and 
Targeted Refusal tasks (Llama-2-7B).
'''

class AttentionDifferenceAnalyzer:
    def __init__(self, config: SafetyNetConfig, model_name, args):
        self.config = config
        self.model_name = model_name
        self.args = args
        
    def load_layer_attention(self, dataset_type: str, layer_idx) -> torch.Tensor:
        """Load all attention scores for a specific layer and dataset type"""
        if self.args.model_type == "vanilla":
            layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type}/layer_{layer_idx}"
        else:
            layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type}/layer_{layer_idx}"
            
        batch_files = glob.glob(f"{layer_dir}/batch_*_qk_scores.pkl")
        
        all_attention = []
        for file_path in sorted(batch_files):
            with open(file_path, "rb") as f:
                batch_attention = pkl.load(f)
                all_attention.append(batch_attention)
        
        return torch.cat(all_attention, dim=0)

    def cosine_sim(self, layer_a, layer_b) -> float:
        pass
        
    
    def analyse_layer(self, layers):
        
        Layers = {'normal': {}, 'harmful': {}}
        datasets = ['normal', 'harmful']
        
        for idx, layer_idx in enumerate(layers):
            for dataset_type in datasets:
                Layers[idx][dataset_type] = self.load_layer_attention(dataset_type, layer_idx)
        
        set_0_normal = self.cosine_sim(Layers[0]['normal'],
                        Layers[1],['normal']
                        )
        
        set_1_normal = self.cosine_sim(Layers[1]['normal'],
                        Layers[2],['normal']
                        )
        
        set_0_harmful = self.cosine_sim(Layers[0]['harmful'],
                        Layers[1],['harmful']
                        )
        
        set_1_normal = self.cosine_sim(Layers[1]['harmful'],
                        Layers[2],['harmful']
                        )

        
        
        
    


class Main:
    @staticmethod
    def main(args):
        config = SafetyNetConfig()
        analyzer = AttentionDifferenceAnalyzer(config, args)
        
        layers = [config.discriminative_layer-1, config.discriminative_layer, config.discriminative_layer+1]
    
        diff = analyzer.analyze_layer(layers)
        analyzer.save_results(layers, diff)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", required=True, help="Model name")
    parser.add_argument("--model_type", "-mt", required=True, help="Model Type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    Main.main(args)