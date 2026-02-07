from utils import torch, np, os, json, gc, glob, pkl, argparse, tqdm, F, accuracy_score, precision_score, recall_score, f1_score
from src.configs.safetynet_config import SafetyNetConfig
from src.configs.spylab_model_config import spylab_create_config
from src.configs.anthropic_model_config import anthropic_create_config


class AttentionDifferenceAnalyzer:
    def __init__(self, config: SafetyNetConfig, model_name, args):
        self.config = config
        self.model_name = model_name
        self.args = args
        self.threshold_scale = 2.0
        
    def load_layer_attention(self, dataset_type: str, layer_idx) -> torch.Tensor:
        """Modified to use half precision"""
        
        if self.args.dataset == "mad":
            if self.args.model_type == "vanilla":
                layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{self.args.model_type}/{dataset_type}/layer_{layer_idx}"
                print(f"PATH: {layer_dir}")
            elif self.args.model_type == "backdoored" or self.args.model_type == "obfuscated_sim":
                layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type}/layer_{layer_idx}"
 
        elif self.args.dataset == "spylab":
            if self.args.model_type == "vanilla":
                layer_dir = f"{self.config.scratch_dir}/{self.args.dataset}/{self.model_name}/{self.args.model_type}/{dataset_type}/layer_{layer_idx}"
                print(f"PATH: {layer_dir}")
            elif self.args.model_type == "backdoored" or self.args.model_type == "obfuscated_sim":
                layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type}/layer_{layer_idx}"

        elif self.args.dataset == "anthropic":
            layer_dir = f"{self.config.scratch_dir}/{self.args.dataset}/{self.model_name}/{self.args.model_type}/{dataset_type}/layer_{layer_idx}"

            

        batch_files = glob.glob(f"{layer_dir}/batch_*_qk_scores.pkl")
        for file in batch_files[:5]:  # Show first 5
            print(f"  - {file}")
        if not batch_files:
            print(f"No files found matching pattern in: {layer_dir}")
    
            
        all_attention = []
        for file_path in tqdm(sorted(batch_files), desc=f"Loading {dataset_type} L{layer_idx}"):
            with open(file_path, "rb") as f:
                data = pkl.load(f)
                # Convert to half precision to save memory
                if data.dtype == torch.float32:
                    data = data.half()
                all_attention.append(data)
        
        return torch.cat(all_attention, dim=0)

    def _cosine_sim_batched_streaming(self, dataset_type_a: str, dataset_type_b: str, 
                                    layer_a_idx: int, layer_b_idx: int, 
                                    batch_size=100) -> float:
        """Compute cosine similarity by streaming data batch by batch"""
        # Get file paths
        if self.args.dataset == "mad":
            if self.args.model_type == "vanilla":
                layer_a_dir = f"{self.config.scratch_dir}/{self.model_name}/{self.args.model_type}/{dataset_type_a}/layer_{layer_a_idx}"
                layer_b_dir = f"{self.config.scratch_dir}/{self.model_name}/{self.args.model_type}/{dataset_type_b}/layer_{layer_b_idx}"
            else:
                layer_a_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type_a}/layer_{layer_a_idx}"
                layer_b_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type_b}/layer_{layer_b_idx}"

        elif self.args.dataset == "anthropic":
            layer_a_dir = f"{self.config.scratch_dir}/{self.args.dataset}/{self.model_name}/{self.args.model_type}/{dataset_type_a}/layer_{layer_a_idx}"
            layer_b_dir = f"{self.config.scratch_dir}/{self.args.dataset}/{self.model_name}/{self.args.model_type}/{dataset_type_b}/layer_{layer_b_idx}"

        
        elif self.args.dataset == "spylab":
            layer_a_dir = f"{self.config.scratch_dir}/{self.args.dataset}/{self.model_name}/{self.args.model_type}/{dataset_type_a}/layer_{layer_a_idx}"
            layer_b_dir = f"{self.config.scratch_dir}/{self.args.dataset}/{self.model_name}/{self.args.model_type}/{dataset_type_b}/layer_{layer_b_idx}"
        
        batch_files_a = sorted(glob.glob(f"{layer_a_dir}/batch_*_qk_scores.pkl"))
        batch_files_b = sorted(glob.glob(f"{layer_b_dir}/batch_*_qk_scores.pkl"))
        
        all_similarities = []
        
        # Process file by file
        for file_a, file_b in tqdm(zip(batch_files_a, batch_files_b), 
                               total=min(len(batch_files_a), len(batch_files_b)),
                               desc=f"Processing {dataset_type_a} L{layer_a_idx}→L{layer_b_idx}"):
        # Load one batch
            with open(file_a, "rb") as f:
                data_a = pkl.load(f).float()
            with open(file_b, "rb") as f:
                data_b = pkl.load(f).float()
            
            # Reshape
            data_a = data_a.reshape(data_a.shape[0], -1).cpu()
            data_b = data_b.reshape(data_b.shape[0], -1).cpu()
            
            # Compute cosine similarity for this batch
            min_samples = min(data_a.shape[0], data_b.shape[0])
            data_a = data_a[:min_samples]
            data_b = data_b[:min_samples]
            
            cos_sim = F.cosine_similarity(data_a, data_b, dim=1)
            all_similarities.extend(cos_sim.tolist())
            
            # Delete immediately
            del data_a, data_b, cos_sim
            gc.collect()
        
        return all_similarities

    def analyze_layer(self, layers):
        """Modified to use streaming approach"""
        print(f"Analyzing layers: {layers}")
        
        results = {'layers': layers,'normal': [], 'harmful': [], 'normal_mean': [], 'harmful_mean': []}
        
        for i in range(len(layers) - 1):
            curr, next_layer = layers[i], layers[i + 1]
            print(f"\nProcessing L{curr} -> L{next_layer}")
            
            # Compute similarities directly without loading all data
            normal_sim = self._cosine_sim_batched_streaming('normal', 'normal', curr, next_layer)
            results['normal'].append(normal_sim)
            results['normal_mean'].append(np.mean(normal_sim))  # np.mean handles lists directly

            harmful_sim = self._cosine_sim_batched_streaming('harmful', 'harmful', curr, next_layer)
            results['harmful'].append(harmful_sim)
            results['harmful_mean'].append(np.mean(harmful_sim))  # np.mean handles lists directly
            
            normal_sim_array = np.array(normal_sim)
            self.fit(normal_sim_array)  # This sets positive_threshold and negative_threshold    
            
            normal_labels = np.zeros(len(normal_sim))
            harmful_labels = np.ones(len(harmful_sim))

            all_distances = np.concatenate([normal_sim, harmful_sim])
            all_labels = np.concatenate([normal_labels, harmful_labels])

            acc, preds = self.calculate_accuracy(all_distances, all_labels)

            # Calculate metrics for harmful data
            harmful_acc = accuracy_score(harmful_labels, preds[len(normal_sim):])
            harmful_prec = precision_score(harmful_labels, preds[len(normal_sim):])
            harmful_rec = recall_score(harmful_labels, preds[len(normal_sim):])
            harmful_f1 = f1_score(harmful_labels, preds[len(normal_sim):])

            # Calculate metrics for normal data
            normal_acc = accuracy_score(normal_labels, preds[:len(normal_sim)])
            normal_prec = precision_score(normal_labels, preds[:len(normal_sim)], zero_division=0)
            normal_rec = recall_score(normal_labels, preds[:len(normal_sim)], zero_division=0)
            normal_f1 = f1_score(normal_labels, preds[:len(normal_sim)], zero_division=0)

            # Print results
            print("="*50)
            print(f"🚨 DETECTION RESULTS for HARMFUL DATA of Layer {layers[i]} and {layers[i+1]} 🚨")
            print("="*50)
            print(f"🎯 Accuracy:  {harmful_acc:.4f}")
            print(f"🔍 Precision: {harmful_prec:.4f}")
            print(f"📊 Recall:    {harmful_rec:.4f}")
            print(f"⚡ F1-Score:  {harmful_f1:.4f}")
            print("="*50)

            print("\n" + "="*50)
            print(f"✅ DETECTION RESULTS for NORMAL DATA of Layer {layers[i]} and {layers[i+1]} ✅")
            print("="*50)
            print(f"🎯 Accuracy:  {normal_acc:.4f}")
            print(f"🔍 Precision: {normal_prec:.4f}")
            print(f"📊 Recall:    {normal_rec:.4f}")
            print(f"⚡ F1-Score:  {normal_f1:.4f}")
            print("="*50)
            
            # print(f"L{curr}->L{next_layer}: Normal={normal_sim:.4f}, Harmful={harmful_sim:.4f}, Diff={abs(normal_sim-harmful_sim):.4f}")
            
            gc.collect()
        
        return results
    
    def save_results(self, layers, results):
        
        os.makedirs(f"{self.config.output_dir}/{self.args.model_type}", exist_ok=True)
        with open(f"{self.config.output_dir}/{self.args.model_type}/cosine_analysis.json", 'w') as f:
            json.dump({
                'layers': layers,
                'normal': results['normal'],
                'harmful': results['harmful'],
                'normal_mean': results['normal_mean'],
                'harmful_mean': results['harmful_mean'],
                'differences': [abs(n-h) for n, h in zip(results['normal_mean'], results['harmful_mean'])]
            }, f, indent=2)
    
        
    
    def fit(self, train_distances, train_labels=None):
        """Fit thresholds using training distances"""
        self.positive_threshold = train_distances.mean() + self.threshold_scale * train_distances.std()
        self.negative_threshold = train_distances.mean() - self.threshold_scale * train_distances.std()
        self.is_fitted = True
        
        return train_distances


    def calculate_accuracy(self, distances, true_labels):
        """Calculate accuracy using the fitted thresholds"""
        if not self.is_fitted:
            raise ValueError("Must fit before calculating accuracy")
        
        pred_labels = ((distances > self.positive_threshold) | 
                    (distances < self.negative_threshold)).astype(int)
        accuracy = (pred_labels == true_labels).mean()
        
        return accuracy, pred_labels


class Main:
    @staticmethod
    def main(args):
        if args.dataset == "mad":
            config = SafetyNetConfig(args.model_name)
        elif args.dataset == "spylab":
            config = spylab_create_config(args.model_name)
        elif args.dataset == "anthropic":
            config = anthropic_create_config(args.model_name)
        analyzer = AttentionDifferenceAnalyzer(config, args.model_name, args)   
        layers = [config.discriminative_layer-1, config.discriminative_layer, config.discriminative_layer+1]
        results = analyzer.analyze_layer(layers)
        analyzer.save_results(layers, results)

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", required=True)
    parser.add_argument("--model_type", "-mt", required=True)
    parser.add_argument("--dataset", required=True, help="mad or spylab")
    return parser.parse_args()

if __name__ == "__main__":
    
    args = parser()
    print(vars(args))
    Main.main(args)