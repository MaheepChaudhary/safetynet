from utils import *
from src.configs.model_configs import AnalysisConfig
from utils.safetynet.detectors import VariationalAutoencoder, Autoencoder
from src.configs.safetynet_config import SafetyNetConfig

class Attention_DataProcessing:
    def __init__(self, model_name: str, config: SafetyNetConfig, layer_idx):
        self.model_name = model_name
        self.config = config
        self.layer_idx = layer_idx
        
    def load_layer_attention(self, dataset_type: str) -> torch.Tensor:
            """Load all attention scores for a specific layer and dataset type"""
            layer_dir = f"{self.config.scratch_dir}/{self.model_name}/{dataset_type}/layer_{self.layer_idx}"
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
            recon, mu, logvar = output
            recon_loss = self.loss(recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + self.safe_config.beta * kl_loss
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
        
        while iter_idx * self.config.batch_size < len(data):
            current_samples = iter_idx*self.config.batch_size
            leftover_samples = len(data) - current_samples
            current_batch_size = min(self.config.batch_size, leftover_samples)
            future_samples = current_samples + current_batch_size 
            batch = data[current_samples:future_samples]
            batch = batch.to(self.device).float()
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.compute_loss(batch, output)
            loss.backward()
            self.optimizer.step()
            iter_idx+=1
            
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
        y_pred = np.concatenate([
            (np.array(normal_losses) > safe_config.threshold).astype(int),
            (np.array(harmful_losses) > safe_config.threshold).astype(int)
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

