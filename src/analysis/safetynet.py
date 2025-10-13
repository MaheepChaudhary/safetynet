from utils import *
from src.configs.model_configs import AnalysisConfig
from utils.safetynet.vae_ae_train import Attention_DataProcessing, Train, Test, Detector_Stats
from utils.safetynet.detectors import PCA, Mahalanobis, Beatrix
from src.configs.safetynet_config import SafetyNetConfig
from utils.visualisation.plot_violin_classification import *  # Import the visualization class


#TODO: We should initially find out which method is the best, even for detection, i.e. using softmax 
#TODO:      probabilty distribution or using raw QKs?

class Monitor:
    
    def __init__(self, args, config: SafetyNetConfig):
        '''
        meta detector is ae, vae, or pca. 
        '''
        self.args = args
        self.config = config
        
        # Initialize data processor
        self.data_processor = Attention_DataProcessing(args.model_name, 
                                                       config, 
                                                       args.layer_idx, 
                                                       model_type = args.model_type
                                                       )
        
        # Initialize trainer/detector based on model type
        detector_choice = self.args.detector
        
        
        if detector_choice == 'pca':
            # For PCA, initialize the detector directly (no training needed)
            self.pca_detector = PCA(n_components=self.config.n_components, 
                                   threshold_scale=self.config.threshold_scale)
            self.trainer = None  # PCA doesn't need a trainer
        
        elif detector_choice == "beatrix":
            self.beatrix_detector = Beatrix()
        
        elif detector_choice == 'mahalanobis':  # ADD THIS
            self.mahalanobis_detector = Mahalanobis(epsilon=self.config.epsilon, 
                                                   threshold_scale=self.config.threshold_scale)
            self.trainer = None
        
        
        else:
            self.trainer = Train(args, config, detector_choice)
        
        
        # Initialize stats calculator
        self.stats = Detector_Stats()
        
    def make_json_serializable(self, obj):
        """Convert numpy arrays and tensors to JSON-serializable formats"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj
    
    def forward(self):
        """Main training and evaluation pipeline"""
        # Load data using existing data processor
        normal_data, harmful_data = self.data_processor.forward()
        print(f"Normal data: {normal_data.shape}, Harmful data: {harmful_data.shape}")
        
        data_mean = normal_data.mean(dim=0, keepdim=True)
        data_std = normal_data.std(dim=0, keepdim=True) + 1e-8  # Prevent division by zero
        
        # Normalize ALL data using training statistics
        normal_data = (normal_data - data_mean) / data_std
        harmful_data = (harmful_data - data_mean) / data_std
        
        # Split normal data for train/val
        val_size = min(1000, len(normal_data) // 5)
        val_data = normal_data[:val_size]
        train_data = normal_data[val_size:]
        
        print(f"Training on {len(train_data)} normal samples")
        print(f"Validation: {len(val_data)} normal, {len(harmful_data)} harmful samples")
        
        # Handle PCA vs neural network training
        if self.args.detector == 'pca':
            print(f"\nFitting PCA detector...")
            self.pca_detector.fit(train_data)
            
            # For PCA, get distances instead of losses
            normal_distances, normal_labels, _ = self.pca_detector(train_data)  # Not used, just for consistency
            val_distances, val_labels, _ = self.pca_detector(val_data)
            harmful_distances, harmful_labels, _ = self.pca_detector(harmful_data)
            
            # Convert distances to "losses" for compatibility with existing code
            # Fix: Handle device properly when converting to numpy
            normal_losses = normal_distances.cpu().numpy()  # Add .cpu()
            val_losses = val_distances.cpu().numpy()     # Add .cpu()
            harmful_losses = harmful_distances.cpu().numpy()  # Add .cpu()
            
            threshold = self.pca_detector.threshold
        
        elif self.args.detector == "beatrix":
            # Fit on subset and immediately free training data
            _ = self.beatrix_detector.fit(train_data)
            
            # Process normal data
            normal_distances, _ = self.beatrix_detector.forward(train_data)
            normal_losses = normal_distances.cpu().numpy()
            del normal_distances
            del train_data  # Free large training data after use
            
            # Process validation data
            val_distances, _ = self.beatrix_detector.forward(val_data)
            val_losses = val_distances.cpu().numpy()
            del val_distances
            
            # Process harmful data
            harmful_distances, _ = self.beatrix_detector.forward(harmful_data)
            harmful_losses = harmful_distances.cpu().numpy()
            del harmful_distances
            
            # Get threshold
            threshold = self.beatrix_detector.threshold
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        
        elif self.args.detector == 'mahalanobis':  # ADD THIS SECTION
            print(f"\nFitting Mahalanobis detector...")
            self.mahalanobis_detector.fit(train_data)
            
            normal_distances, normal_labels, _ = self.mahalanobis_detector(train_data)
            val_distances, val_labels, _ = self.mahalanobis_detector(val_data)
            harmful_distances, harmful_labels, _ = self.mahalanobis_detector(harmful_data)
            
            # Convert distances to "losses" for compatibility
            normal_losses = normal_distances.cpu().numpy()
            val_losses = val_distances.cpu().numpy()
            harmful_losses = harmful_distances.cpu().numpy()
            
            threshold = self.mahalanobis_detector.threshold
        
        
        else:
            # Original neural network training
            print(f"\nTraining {self.config.model_name}...")
            for epoch in tqdm(range(self.config.epochs)):
                train_loss = self.trainer.forward(train_data)
                # if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {train_loss:.4f}")
            
            # Save model
            os.makedirs(self.config.output_dir, exist_ok=True)
            torch.save(self.trainer.model.state_dict(), 
                      f'{self.config.output_dir}/{self.args.model_type}_{self.args.detector}_detector.pth')
            
            # Evaluation using existing test class
            print("\nEvaluating...")
            normal_tester = Test(train_data, self.config, self.args.detector)
            val_tester = Test(val_data, self.config, self.args.detector)
            harmful_tester = Test(harmful_data, self.config, self.args.detector)
            
            normal_losses = normal_tester.forward()
            val_losses = val_tester.forward()
            harmful_losses = harmful_tester.forward()
            
            # Set threshold (mean + 2*std of validation losses)
            threshold = np.mean(val_losses) + 2 * np.std(val_losses)
        
        self.config.threshold = threshold  # Update config with computed threshold
        # print(f"Detection threshold: {threshold:.4f}")
        
        # Compute metrics using existing stats calculator
        results = self.stats.compute_metrics(val_losses, harmful_losses, self.config)
        # normal_metrics = self.stats.compute_metrics(val_losses, normal_losses, self.config)
    
        
        # Print resultsprint("\n" + "="*50)
        # print("🚨 DETECTION RESULTS for HARMFUL DATA 🚨")
        # print("="*50)
        # print(f"🎯 Accuracy:  {harmful_metrics['accuracy']:.4f}")
        # print(f"🔍 Precision: {harmful_metrics['precision']:.4f}")
        # print(f"📊 Recall:    {harmful_metrics['recall']:.4f}")
        # print(f"⚡ F1-Score:  {harmful_metrics['f1']:.4f}")
        # print("="*50)
        print(results)
        
        # print("\n" + "="*50)
        # print("✅ DETECTION RESULTS for NORMAL DATA ✅")
        # print("="*50)
        # print(f"🎯 Accuracy:  {normal_metrics['accuracy']:.4f}")
        # print(f"🔍 Precision: {normal_metrics['precision']:.4f}")
        # print(f"📊 Recall:    {normal_metrics['recall']:.4f}")
        # print(f"⚡ F1-Score:  {normal_metrics['f1']:.4f}")
        # print("="*50)
        
        # save_path = f"{self.config.output_dir}/{self.args.detector}_layer_{self.args.layer_idx}_{self.args.model_type}"

        # loss_data_dict = {
        #     "normal_losses": self.make_json_serializable(normal_losses),
        #     "val_losses": self.make_json_serializable(val_losses), 
        #     "harmful_losses": self.make_json_serializable(harmful_losses),
        #     "threshold": self.make_json_serializable(threshold)
        #     }     
          
        # data_save_path = f"utils/data/{self.args.model_name}/{self.args.model_type}_{self.args.detector}_loss/"
        # os.makedirs(data_save_path, exist_ok=True)
        # with open(f"{data_save_path}/layer_{self.args.layer_idx}_{self.args.model_type}_{self.args.detector}_loss.json", "w") as f: json.dump(loss_data_dict, f, indent=2)
        
        return results
    


def main():
    parser = argparse.ArgumentParser(description='Attention-based Outlier Detection')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name for attention data loading')
    parser.add_argument('--detector', type=str, required=True, 
                       choices=['ae', 'vae', 'pca', 'mahalanobis', 'beatrix', 'crow'],  # Add 'pca' here
                       help='choice: ae, vae, pca, or mahalanobis')
    parser.add_argument("--layer_idx", type=int, required=True,
                        help="which layer do you want to monitor?")
    parser.add_argument("--model_type", type=str, required=True,
                        help="is the model backdoor, vanilla, obfuscated_sim, obfuscated_ae")
    
    # python -m src.analysis.safetynet --layer_idx 15 --model_name llama2 --model_type vae
    
    args = parser.parse_args()
    
    print(vars(args))
    
    config = SafetyNetConfig(args.model_name)
    
    # Create monitor and run
    monitor = Monitor(args, config)
    metrics = monitor.forward()
    
    return metrics


if __name__ == '__main__':
    main()