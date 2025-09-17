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
        
        


class Beatrix(nn.Module):
    def __init__(self, mad_scale=3.0, batch_size=32, cpu_only_mode=True, low_precision=True, power=2, max_features=1000):
        super().__init__()
        self.mad_scale = mad_scale
        self.batch_size = batch_size
        self.cpu_only_mode = cpu_only_mode
        self.low_precision = low_precision
        self.power = power
        self.max_features = max_features  # NEW: dimensionality limit
        self.pca = None  # NEW: PCA reducer
        self.device = torch.device('cpu' if cpu_only_mode else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.register_buffer('_device_dummy', torch.tensor(0.0))
        self.median = None
        self.mad = None
        self.threshold = None
        self.is_fitted = False
    
    def _reduce_dims(self, x, fit=False):  # NEW METHOD
        """Reduce dimensions if needed"""
        if x.shape[1] <= self.max_features:
            return x
        
        x_np = self._to_numpy(x) if torch.is_tensor(x) else x.reshape(x.shape[0], -1)
        
        if fit:
            print(f"Reducing dims from {x_np.shape[1]} to {self.max_features}")
            self.pca = PCA(n_components=self.max_features)
            x_reduced = self.pca.fit_transform(x_np)
        else:
            x_reduced = self.pca.transform(x_np)
        
        return torch.from_numpy(x_reduced).float()
    
    def _to_numpy(self, x):
        return x.view(x.shape[0], -1).cpu().numpy() if torch.is_tensor(x) else x.reshape(x.shape[0], -1)
    
    def _to_tensor(self, arr, device):
        return torch.from_numpy(arr).float().to(device)
    
    def _compute_gram_features(self, activation):
        # NEW: Apply dimensionality reduction first
        activation = self._reduce_dims(activation, fit=not self.is_fitted)
        
        if not torch.is_tensor(activation):
            dtype = torch.float16 if self.low_precision else torch.float32
            activation = torch.tensor(activation, dtype=dtype)
            
        num_samples = activation.shape[0]
        gram_results = []
        
        for i in tqdm(range(0, num_samples, self.batch_size)):
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            end_idx = min(i + self.batch_size, num_samples)
            batch = activation[i:end_idx].to(self.device)
            if self.low_precision and batch.dtype != torch.float16: batch = batch.half()
            
            powered_batch = batch ** self.power
            
            for j in range(powered_batch.shape[0]):
                feature_flat = powered_batch[j:j+1].reshape(1, -1)
                gram = torch.matmul(feature_flat.t(), feature_flat)
                gram_results.append(gram.cpu())
                del gram
                gc.collect()
            
            del batch, powered_batch
        
        grams = torch.stack(gram_results)
        grams = grams.sign() * torch.abs(grams) ** (1.0 / self.power)
        
        n = grams.size(-1)
        triu_indices = torch.triu_indices(n, n)
        gram_vectors = []
        
        for i in range(0, grams.shape[0], 10):
            end = min(i + 10, grams.shape[0])
            batch_gram = grams[i:end]
            batch_vector = batch_gram[..., triu_indices[0], triu_indices[1]]
            batch_vector = torch.nan_to_num(batch_vector, nan=0.0, posinf=0.0, neginf=0.0)
            gram_vectors.append(batch_vector)
        
        gram_vector = torch.cat(gram_vectors, dim=0)
        return gram_vector.to(self.device) if not self.cpu_only_mode else gram_vector
    
    def fit(self, normal_activations):
        device = normal_activations.device if torch.is_tensor(normal_activations) else self._device_dummy.device
        gram_features = self._compute_gram_features(normal_activations)
        features_np = self._to_numpy(gram_features)
        
        self.median = np.median(features_np, axis=0)
        self.mad = np.median(np.abs(features_np - self.median), axis=0)
        scores = np.sum(np.abs(features_np - self.median) / (self.mad + 1e-6), axis=1)
        
        score_median = np.median(scores)
        score_mad = np.median(np.abs(scores - score_median))
        self.threshold = score_median + self.mad_scale * score_mad
        self.is_fitted = True
        
        return self._to_tensor(scores, device)
    
    def forward(self, x):
        if not self.is_fitted: raise ValueError("Must fit before forward pass")
        
        device = x.device if torch.is_tensor(x) else self._device_dummy.device
        gram_features = self._compute_gram_features(x)
        features_np = self._to_numpy(gram_features)
        
        scores = np.sum(np.abs(features_np - self.median) / (self.mad + 1e-6), axis=1)
        labels = (scores > self.threshold).astype(int)
        
        scores_tensor = self._to_tensor(scores, device)
        labels_tensor = torch.from_numpy(labels).long().to(device)
        
        return scores_tensor, labels_tensor