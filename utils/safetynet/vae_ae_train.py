from utils import *
from src.configs.model_configs import AnalysisConfig
from utils.safetynet.detectors import VariationalAutoencoder, Autoencoder
from src.configs.safetynet_config import SafetyNetConfig

class Attention_DataProcessing:
    def __init__(self, model_name: str, config: SafetyNetConfig, layer_idx, args, model_type:str, dataset: str):
        self.model_name = model_name
        self.config = config
        self.layer_idx = layer_idx
        self.model_type = model_type
        self.dataset = dataset
        self.args = args
        
    def load_layer_attention(self, dataset_type: str) -> torch.Tensor:
            """Load all attention scores for a specific layer and dataset type"""
            
            if self.args.dataset == "mad":
                layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{self.model_type}/{dataset_type}/layer_{self.layer_idx}"
            
            elif self.args.dataset == "spylab":
                layer_dir = f"{self.config.scratch_dir}/{self.args.dataset}/{self.model_name}/{self.model_type}/{dataset_type}/layer_{self.layer_idx}"
            
            if not os.path.exists(layer_dir):
                raise FileNotFoundError(f"Layer directory does not exist: {layer_dir}")
            print(f"Layer Dir: 🦠 {layer_dir}")
            batch_files = glob.glob(f"{layer_dir}/batch_*_qk_scores.pkl")

            if len(batch_files) == 0:
                raise FileNotFoundError(
                    f"No attention files found in {layer_dir}\n"
                    f"Expected pattern: batch_*_qk_scores.pkl\n"
                    f"Please run attention extraction first:\n"
                    f"  python -m utils.attn_store --model {self.model_name} --model_type {self.model_type} "
                    f"--dataset_type {dataset_type} --dataset {self.args.dataset}"
                )

            _all_attention = []
            for file_path in sorted(batch_files):
                with open(file_path, "rb") as f:
                    batch_attention = pkl.load(f)
                    _all_attention.append(batch_attention)

            all_attention = torch.cat(_all_attention, dim=0).mean(dim = 1).flatten(start_dim=1)
            return all_attention
        
    def forward(self):
        normal_attention = self.load_layer_attention("normal")
        harmful_attention = self.load_layer_attention("harmful")
        
        return normal_attention, harmful_attention

class BaseClass:    
    
    def __init__(self, safe_config: SafetyNetConfig):
        self.loss = safe_config.criterion
        self.safe_config = safe_config

    def compute_loss(self, x, output):
        """Compute reconstruction loss (+ KL for VAE)"""
        if isinstance(output, tuple):  # VAE case
            # In your VAE loss function, replace with this heavily instrumented version:
            recon, mu, logvar = output
            recon_loss = self.loss(recon, x)
            mu_squared = mu.pow(2)
            logvar_exp = logvar.exp()
            kl_loss = -0.5 * torch.sum(1 + logvar - mu_squared - logvar_exp)
            total_loss = recon_loss + self.safe_config.beta * kl_loss
            return total_loss
        else:  # Standard AE
            return self.loss(output, x)
    
    
class Train(BaseClass):
    def __init__(self, 
                 args, 
                 config: SafetyNetConfig, 
                 detector_choice: str
                 ):
        super().__init__(config)
        self.args = args
        self.device = torch.device(config.device)
        self.config = config
        
        if detector_choice == "ae":
            self.model = Autoencoder(self.config.qk_dim).to(config.device)
        
        elif detector_choice == "vae":
            self.model = VariationalAutoencoder(self.config.qk_dim).to(config.device)
        
        self.optimizer = self.config.optim(self.model.parameters())
    
    

    def forward(self, data):
        """Train for one epoch"""
        self.model.train()
        losses = []
        current_batch_size = self.config.batch_size
        iter_idx = 0
        
        for i in range(0, len(data), self.config.batch_size):
            batch = data[i:i + self.config.batch_size]
            batch = batch.to(self.device).float()
            
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.compute_loss(batch, output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
            self.optimizer.step()
            
            losses.append(loss.item())
            
        return np.mean(losses)
    
    

class Test(BaseClass):
    
    def __init__(self, test_data, config: SafetyNetConfig, detector_choice, args):
        super().__init__(config)
        self.test_data = test_data
        self.config = config
        self.detector_choice = detector_choice
        self.args = args
                
        if detector_choice == "ae":
            self.model = Autoencoder(self.config.qk_dim).to(config.device)
        
        elif detector_choice == "vae":
            self.model = VariationalAutoencoder(self.config.qk_dim).to(config.device)
        
    def _load_model(self):
        model_path = f'{self.config.output_dir}/{self.args.model_type}_{self.detector_choice}_detector.pth'
        state_dict = torch.load(model_path)
        
        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)
        return self.model
        
    def forward(self):
        self._load_model()
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for i in range(0, len(self.test_data), self.safe_config.batch_size):
                batch = self.test_data[i:i+self.safe_config.batch_size].to(self.safe_config.device).float()
                output = self.model(batch)
                loss = self.compute_loss(batch, output)
                losses.append(loss.item())
        
        return losses
    
    
    
class Detector_Stats:
    
    
    @staticmethod
    def compute_comprehensive_metrics(train_losses, val_losses, harmful_losses):
        """
        Compute all metrics including AUROC, accuracy, precision, recall, F1
        
        Args:
            train_losses: losses from training normal data
            val_losses: losses from validation normal data  
            harmful_losses: losses from harmful data
            
        Returns:
            Dictionary with all metrics
        """
        # Convert to numpy if needed
        if isinstance(train_losses, torch.Tensor):
            train_losses = train_losses.cpu().numpy()
        if isinstance(val_losses, torch.Tensor):
            val_losses = val_losses.cpu().numpy()
        if isinstance(harmful_losses, torch.Tensor):
            harmful_losses = harmful_losses.cpu().numpy()
            
        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)
        harmful_losses = np.array(harmful_losses)
        
        # Convert to arrays and ensure they're at least 1D
        train_losses = np.atleast_1d(np.array(train_losses))
        val_losses = np.atleast_1d(np.array(val_losses))
        harmful_losses = np.atleast_1d(np.array(harmful_losses))
        
        # Flatten in case of multi-dimensional arrays
        train_losses = train_losses.flatten()
        val_losses = val_losses.flatten()
        harmful_losses = harmful_losses.flatten()
        
        # Calculate thresholds based on training data
        threshold_upper = np.mean(val_losses) + 2 * np.std(val_losses)
        threshold_lower = np.mean(val_losses) - 2 * np.std(val_losses)
        
        # ========== OVERALL METRICS (Val + Harmful) ==========
        # Combine validation (normal) and harmful data
        all_losses = np.concatenate([train_losses, harmful_losses])
        y_true_overall = np.concatenate([
            np.zeros(len(train_losses)),  # 0 for normal
            np.ones(len(harmful_losses))  # 1 for harmful
        ])
        
        # Predictions using threshold
        y_pred_overall = ((all_losses > threshold_upper) | (all_losses < threshold_lower)).astype(int)
        
        # Overall confusion matrix
        cm_overall = confusion_matrix(y_true_overall, y_pred_overall)
        tn_overall, fp_overall, fn_overall, tp_overall = cm_overall.ravel()
        
        # Overall metrics
        overall_accuracy = (tp_overall + tn_overall) / (tp_overall + tn_overall + fp_overall + fn_overall)
        overall_precision = tp_overall / (tp_overall + fp_overall) if (tp_overall + fp_overall) > 0 else 0
        overall_recall = tp_overall / (tp_overall + fn_overall) if (tp_overall + fn_overall) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # ========== TRAIN ACCURACY (on training data) ==========
        train_pred = ((train_losses > threshold_upper) | (train_losses < threshold_lower)).astype(int)
        # All training samples are normal (label 0), so correctly predicted as normal (pred 0) = accurate
        train_accuracy = np.mean(train_pred == 0)
        
        # ========== HARMFUL-SPECIFIC METRICS ==========
        # Only evaluate on harmful data
        harmful_pred = ((harmful_losses > threshold_upper) | (harmful_losses < threshold_lower)).astype(int)
        y_true_harmful = np.ones(len(harmful_losses))  # All harmful samples should be labeled 1
        
        # Harmful confusion matrix
        cm_harmful = confusion_matrix(y_true_harmful, harmful_pred, labels=[0, 1])
        # For harmful-only: tn=0 (no true negatives), fp=0 (no false positives in harmful data)
        # fn = harmful samples predicted as normal (0)
        # tp = harmful samples predicted as harmful (1)
        harmful_tp = np.sum(harmful_pred == 1)
        harmful_fn = np.sum(harmful_pred == 0)
        
        harmful_accuracy = harmful_tp / len(harmful_losses) if len(harmful_losses) > 0 else 0
        harmful_precision = 1.0 if harmful_tp > 0 else 0  # No false positives in harmful-only data
        harmful_recall = harmful_tp / (harmful_tp + harmful_fn) if (harmful_tp + harmful_fn) > 0 else 0
        harmful_f1 = 2 * harmful_precision * harmful_recall / (harmful_precision + harmful_recall) if (harmful_precision + harmful_recall) > 0 else 0
        
        # ========== AUROC ==========
        # AUROC using losses as scores (higher loss = more likely harmful)
        auroc = roc_auc_score(y_true_overall, all_losses)
        
        return {
            # AUROC
            'auroc': auroc,
            
            # Overall metrics (validation + harmful)
            'overall_accuracy': overall_accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            
            # Training accuracy
            'train_accuracy': train_accuracy,
            
            # Harmful-specific metrics
            'harmful_accuracy': harmful_accuracy,
            'harmful_precision': harmful_precision,
            'harmful_recall': harmful_recall,
            'harmful_f1': harmful_f1,
            
            # Thresholds
            'threshold_lower': threshold_lower,
            'threshold_upper': threshold_upper,
            
            # Confusion matrices (for debugging)
            'confusion_matrix_overall': cm_overall,
            'tp_overall': tp_overall,
            'fp_overall': fp_overall,
            'tn_overall': tn_overall,
            'fn_overall': fn_overall
        }
