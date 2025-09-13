from utils import *

#TODO: include all the other detectors like pca, and others here.

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(1024, latent_dim)
        self.logvar_layer = nn.Linear(1024, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        return reconstructed, mu, logvar


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )
    
    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(encoded)
        
        return reconstructed



class PCA(nn.Module):
    def __init__(self, n_components=100, threshold_scale=0.5):
        super().__init__()
        self.n_components = n_components
        self.threshold_scale = threshold_scale
        self.pca = SklearnPCA(n_components=n_components)
        self.normal_mean = None
        self.threshold = None
        self.is_fitted = False
        self.register_buffer('_device_dummy', torch.tensor(0.0))
    
    def _to_numpy(self, x):
        return x.view(x.shape[0], -1).cpu().numpy() if torch.is_tensor(x) else x.reshape(x.shape[0], -1)
    
    def _to_tensor(self, arr, device):
        return torch.from_numpy(arr).float().to(device)
    
    def fit(self, normal_vectors):
        device = normal_vectors.device if torch.is_tensor(normal_vectors) else self._device_dummy.device
        x_np = self._to_numpy(normal_vectors)
        
        normal_pca = self.pca.fit_transform(x_np)
        self.normal_mean = normal_pca.mean(axis=0)
        distances = np.linalg.norm(normal_pca - self.normal_mean, axis=1)
        self.threshold = distances.mean() + self.threshold_scale * distances.std()
        self.is_fitted = True
        
        return self._to_tensor(distances, device)
    
    def forward(self, x):
        if not self.is_fitted:
            raise ValueError("Must fit before forward pass")
        
        device = x.device if torch.is_tensor(x) else self._device_dummy.device
        x_np = self._to_numpy(x)
        
        x_pca = self.pca.transform(x_np)
        distances = np.linalg.norm(x_pca - self.normal_mean, axis=1)
        labels = (distances > self.threshold).astype(int)
        
        return (self._to_tensor(distances, device), 
                torch.from_numpy(labels).long().to(device), 
                self._to_tensor(x_pca, device))





class Mahalanobis(nn.Module):
    def __init__(self, epsilon=1e-6, threshold_scale=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.threshold_scale = threshold_scale
        self.mu = None
        self.cov_inverse = None
        self.threshold = None
        self.is_fitted = False
        self.register_buffer('_device_dummy', torch.tensor(0.0))
    
    def _to_numpy(self, x):
        return x.view(x.shape[0], -1).cpu().numpy() if torch.is_tensor(x) else x.reshape(x.shape[0], -1)
    
    def _to_tensor(self, arr, device):
        return torch.from_numpy(arr).float().to(device)
    
    def fit(self, normal_vectors):
        device = normal_vectors.device if torch.is_tensor(normal_vectors) else self._device_dummy.device
        x_np = self._to_numpy(normal_vectors)
        
        # Compute mean and covariance
        self.mu = np.mean(x_np, axis=0)
        cov_matrix = np.cov(x_np, rowvar=False)
        cov_matrix_reg = cov_matrix + self.epsilon * np.eye(cov_matrix.shape[0])
        
        try:
            self.cov_inverse = np.linalg.inv(cov_matrix_reg)
        except np.linalg.LinAlgError:
            cov_matrix_reg = cov_matrix + 0.1 * np.eye(cov_matrix.shape[0])
            self.cov_inverse = np.linalg.inv(cov_matrix_reg)
        
        # Compute threshold
        train_distances = self._compute_distances_np(x_np)
        self.threshold = train_distances.mean() + self.threshold_scale * train_distances.std()
        self.is_fitted = True
        
        return self._to_tensor(train_distances, device)
    
    def _compute_distances_np(self, data_np):
        centered_data = data_np - self.mu
        mahalanobis_squares = np.sum(np.matmul(centered_data, self.cov_inverse) * centered_data, axis=1)
        return np.sqrt(mahalanobis_squares)
    
    def forward(self, x):
        if not self.is_fitted:
            raise ValueError("Must fit before forward pass")
        
        device = x.device if torch.is_tensor(x) else self._device_dummy.device
        x_np = self._to_numpy(x)
        
        distances = self._compute_distances_np(x_np)
        labels = (distances > self.threshold).astype(int)
        
        return (self._to_tensor(distances, device), 
                torch.from_numpy(labels).long().to(device), 
                self._to_tensor(distances, device))