from utils import *
from src.configs.model_configs import AnalysisConfig
from utils.safetynet.detectors import VariationalAutoencoder, Autoencoder
from src.configs.safetynet_config import SafetyNetConfig

class Attention_DataProcessing:
    def __init__(self, model_name: str, config: SafetyNetConfig, layer_idx, model_type:str):
        self.model_name = model_name
        self.config = config
        self.layer_idx = layer_idx
        self.model_type = model_type
        
    def load_layer_attention(self, dataset_type: str) -> torch.Tensor:
            """Load all attention scores for a specific layer and dataset type"""
            if self.model_type == "backdoored":
                layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type}/layer_{self.layer_idx}"
            else:
                layer_dir = os.path.abspath(f"{self.config.scratch_dir}/{self.model_name}/{self.model_type}/{dataset_type}/layer_{self.layer_idx}")
            if not os.path.exists(layer_dir):
                raise FileNotFoundError(f"Layer directory does not exist: {layer_dir}")
            print(f"Layer Dir: 🦠 {layer_dir}")
            batch_files = glob.glob(f"{layer_dir}/batch_*_qk_scores.pkl")
            
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
    
    def __init__(self, test_data, config: SafetyNetConfig, detector_choice):
        super().__init__(config)
        self.test_data = test_data
        self.config = config
        self.detector_choice = detector_choice
                
        if detector_choice == "ae":
            self.model = Autoencoder(self.config.qk_dim).to(config.device)
        
        elif detector_choice == "vae":
            self.model = VariationalAutoencoder(self.config.qk_dim).to(config.device)
        
    def _load_model(self):
        model_path = f'{self.config.output_dir}/{self.detector_choice}_detector.pth'
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
    def compute_metrics(normal_losses, harmful_losses, safe_config: SafetyNetConfig):
        """Compute detection metrics"""
        # Labels: 0 for normal, 1 for harmful
        y_true = np.concatenate([np.zeros(len(normal_losses)), 
                                np.ones(len(harmful_losses))])
        
        # Predictions: loss > threshold → harmful (1)
        threshold = np.mean(normal_losses) + 2 * np.std(normal_losses)
        threshold_neg = np.mean(normal_losses) - 2 * np.std(normal_losses)
        y_pred = np.concatenate([
            ((np.array(normal_losses) > threshold) | (np.array(normal_losses) < threshold_neg)).astype(int),
            ((np.array(harmful_losses) > threshold) | (np.array(harmful_losses) < threshold_neg)).astype(int)
        ])
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }

