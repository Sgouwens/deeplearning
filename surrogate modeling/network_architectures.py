import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Define base class containing the forward functions.
class BaseSurrogate(nn.Module):
    
    def __init__(self):
        super(BaseSurrogate, self).__init__()

    def forward(self, x):
        return self.network(x) + x  # Residual connection
    
    def forward_k_steps(self, x, k=1):
        for _ in range(k):
            x = self.forward(x)
        return x

### Simplest linear network (but with many parameters..) only works with 100 x 100 simulations
class WaveEquationSurrogateLinear(nn.Module):
    def __init__(self):
        super(WaveEquationSurrogateLinear, self).__init__()

        self.model_name = "WaveEquationSurrogateLinear"
        
        # A smaller latent space (e.g., 1024-dimensional)
        self.fc1 = nn.Linear(100 * 100, 8)  # Flatten input to smaller latent space
        self.activation = nn.SiLU()  # Activation function
        self.fc2 = nn.Linear(8, 100 * 100)  # Map back to 10000-dimensional space
        
    def forward(self, x):
        # Flatten the input image (batch_size, 100, 100) -> (batch_size, 100*100)
        x = x.view(x.size(0), -1)
        
        # Apply the first fully connected layer with bottleneck
        x = self.activation(self.fc1(x))
        
        # Apply the second fully connected layer
        x = self.fc2(x)
        
        # Reshape to the original image size
        x = x.view(x.size(0), 100, 100)
        
        return x

### Simplest convolutional network (shallow)
class SingleLayerConvolutional(BaseSurrogate):
    # A fully convolutional neural network for solving the 2D wave equation
    def __init__(self):
        super(SingleLayerConvolutional, self).__init__()

        self.model_name = "SingleLayerConvolutional"

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),  # Output: (8, 100, 100)
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0),  # Output: (1, 100, 100)
        )

class MultiLayerConvolutional(BaseSurrogate):
    # A fully convolutional neural network for solving the 2D wave equation
    def __init__(self, convlayers=1):
        super(MultiLayerConvolutional, self).__init__()
        self.model_name = f"{convlayers}LayerConvolutional"
        
        layers = []
        
        layers.extend([
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(8),
            nn.SiLU()
        ])
        
        for _ in range(convlayers - 1):
            layers.extend([
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm2d(8),
                nn.SiLU()
            ])
        
        # Final convolutional layer
        layers.append(nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0))
        
        # Create the Sequential model
        self.network = nn.Sequential(*layers)



class SurrogateEncodeDecode(BaseSurrogate):
    # A fully convolutional neural network for solving the 2D wave equation
    def __init__(self):
        super(SurrogateEncodeDecode, self).__init__()

        self.model_name = "SurrogateEncodeDecode"

        self.network = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),  # Output: (8, 50, 50)
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),  # Output: (16, 25, 25)
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),  # Output: (32, 13, 13)
            nn.BatchNorm2d(32),
            nn.SiLU(),

            # Middle
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: (32, 13, 13)
            nn.BatchNorm2d(32),
            nn.SiLU(),

            # Decoder
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=0),  # Output: (16, 25, 25)
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output: (8, 50, 50)
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output: (1, 100, 100)
            nn.BatchNorm2d(1),
            nn.SiLU(),

            # Final adjustment
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),  # Output: (1, 100, 100)
        )



### Model 3
import torch
import torch.nn as nn

# Define base class containing the forward functions.
class BaseSurrogate(nn.Module):
    
    def __init__(self):
        super(BaseSurrogate, self).__init__()

    def forward(self, x):
        return self.network(x) + x  # Residual connection
    
    def forward_k_steps(self, x, k=1):
        for _ in range(k):
            x = self.forward(x)
        return x

### Simplest linear network (but with many parameters..) only works with 100 x 100 simulations
class WaveEquationSurrogateLinear(nn.Module):
    def __init__(self):
        super(WaveEquationSurrogateLinear, self).__init__()

        self.model_name = "WaveEquationSurrogateLinear"
        
        # A smaller latent space (e.g., 1024-dimensional)
        self.fc1 = nn.Linear(100 * 100, 8)  # Flatten input to smaller latent space
        self.activation = nn.SiLU()  # Activation function
        self.fc2 = nn.Linear(8, 100 * 100)  # Map back to 10000-dimensional space
        
    def forward(self, x):
        # Flatten the input image (batch_size, 100, 100) -> (batch_size, 100*100)
        x = x.view(x.size(0), -1)
        
        # Apply the first fully connected layer with bottleneck
        x = self.activation(self.fc1(x))
        
        # Apply the second fully connected layer
        x = self.fc2(x)
        
        # Reshape to the original image size
        x = x.view(x.size(0), 100, 100)
        
        return x

### Simplest convolutional network (shallow)
class SingleLayerConvolutional(BaseSurrogate):
    # A fully convolutional neural network for solving the 2D wave equation
    def __init__(self):
        super(SingleLayerConvolutional, self).__init__()

        self.model_name = "SingleLayerConvolutional"

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),  # Output: (8, 100, 100)
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0),  # Output: (1, 100, 100)
        )

class MultiLayerConvolutional(BaseSurrogate):
    # A fully convolutional neural network for solving the 2D wave equation
    def __init__(self, convlayers=1):
        super(MultiLayerConvolutional, self).__init__()
        self.model_name = f"{convlayers}LayerConvolutional"
        
        layers = []
        
        layers.extend([
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(8),
            nn.SiLU()
        ])
        
        for _ in range(convlayers - 1):
            layers.extend([
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=1, padding=3),
                nn.BatchNorm2d(8),
                nn.SiLU()
            ])
        
        # Final convolutional layer
        layers.append(nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0))
        
        # Create the Sequential model
        self.network = nn.Sequential(*layers)



class SurrogateEncodeDecode(BaseSurrogate):
    # A fully convolutional neural network for solving the 2D wave equation
    def __init__(self):
        super(SurrogateEncodeDecode, self).__init__()

        self.model_name = "SurrogateEncodeDecode"

        self.network = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),  # Output: (8, 50, 50)
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),  # Output: (16, 25, 25)
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),  # Output: (32, 13, 13)
            nn.BatchNorm2d(32),
            nn.SiLU(),

            # Middle
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: (32, 13, 13)
            nn.BatchNorm2d(32),
            nn.SiLU(),

            # Decoder
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=0),  # Output: (16, 25, 25)
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output: (8, 50, 50)
            nn.BatchNorm2d(8),
            nn.SiLU(),

            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output: (1, 100, 100)
            nn.BatchNorm2d(1),
            nn.SiLU(),

            # Final adjustment
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0),  # Output: (1, 100, 100)
        )



### Model 3
class DoubleConv(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.network(x)
    

class SurrogateUNet(nn.Module):
    def __init__(self):
        super(SurrogateUNet, self).__init__()

        self.model_name = "SurrogateUNet"

        # Encoder
        self.encode1 = DoubleConv(1, 8)
        self.encode1_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode2 = DoubleConv(8, 16)
        self.encode2_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode3 = DoubleConv(16, 32)
        self.encode3_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode4 = DoubleConv(32, 64)
        self.encode4_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(64, 64)

        # Decoder
        self.decode4_transpose = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.decode4 = DoubleConv(128, 32)

        self.decode3_transpose = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, output_padding=1)
        self.decode3 = DoubleConv(64, 16)

        self.decode2_transpose = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.decode2 = DoubleConv(32, 8)

        self.decode1_transpose = nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2)
        self.decode1 = DoubleConv(16, 1)

    def forward(self, x):
        # Encoder
        encode1 = self.encode1(x)
        encode1_maxpool = self.encode1_maxpool(encode1)

        encode2 = self.encode2(encode1_maxpool)
        encode2_maxpool = self.encode2_maxpool(encode2)

        encode3 = self.encode3(encode2_maxpool)
        encode3_maxpool = self.encode3_maxpool(encode3)

        encode4 = self.encode4(encode3_maxpool)
        encode4_maxpool = self.encode4_maxpool(encode4)

        # Bottleneck
        bottleneck = self.bottleneck(encode4_maxpool)

        # Decoder
        decode4 = self.decode4_transpose(bottleneck)
        decode4 = torch.cat([encode4, decode4], dim=1)  # Concatenate along channels
        decode4 = self.decode4(decode4)

        decode3 = self.decode3_transpose(decode4)
        decode3 = torch.cat([encode3, decode3], dim=1)  # Concatenate along channels
        decode3 = self.decode3(decode3)

        decode2 = self.decode2_transpose(decode3)
        decode2 = torch.cat([encode2, decode2], dim=1)  # Concatenate along channels
        decode2 = self.decode2(decode2)

        decode1 = self.decode1_transpose(decode2)
        decode1 = torch.cat([encode1, decode1], dim=1)  # Concatenate along channels
        decode1 = self.decode1(decode1)

        return decode1 + x  # Residual connection
    
# Variational autoencoder prior to 

class VAE(nn.Module):
    # Check out batch normalization
    def __init__(self, latent_dim=128, scale=1):
        super(VAE, self).__init__()
        
        self.scale = scale

        # Encoder: convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, int(2 * self.scale), kernel_size=3, stride=2, padding=1),  # (1, 100, 100) -> (2*scale, 50, 50)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(2 * self.scale), int(4 * self.scale), kernel_size=3, stride=2, padding=1),  # -> (4*scale, 25, 25)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(4 * self.scale), int(8 * self.scale), kernel_size=3, stride=2, padding=1),  # -> (8*scale, 12, 12)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(int(8 * self.scale), int(8 * self.scale), kernel_size=3, stride=2, padding=1),  # -> (8*scale, 7, 7)
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Linear layers to output mean and log variance for latent space
        self.fc1 = nn.Linear(int(8 * self.scale) * 7 * 7, latent_dim)  # Mean
        self.fc2 = nn.Linear(int(8 * self.scale) * 7 * 7, latent_dim)  # Log variance
        
        # Decoder: transpose convolutional layers with scaling factor
        self.fc3 = nn.Linear(latent_dim, int(8 * scale) * 7 * 7)  # To reshape latent space back to (8*scale, 7, 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(8 * self.scale), int(8 * self.scale), kernel_size=3, stride=2, padding=1, output_padding=0),  # -> (8*scale, 12, 12)
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(int(8 * self.scale), int(4 * self.scale), kernel_size=3, stride=2, padding=1, output_padding=0),  # -> (4*scale, 24, 24)
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(int(4 * self.scale), int(2 * self.scale), kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (2*scale, 48, 48)
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(int(2 * self.scale), 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (1, 100, 100)
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.latent_dim = latent_dim

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc1(x)  # Mean
        logvar = self.fc2(x)  # Log variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        x = self.fc3(z)  # (B, latent_dim) -> (B, 8*6*6)
        x = x.view(x.size(0), 8*self.scale, 7, 7)  # Reshape to (B, 8, 6, 6)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)  # Get mean and log variance
        z = self.reparameterize(mu, logvar)  # Sample from the latent space
        reconstructed_x = self.decode(z)  # Decode back to original space
        return reconstructed_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        MSE = F.mse_loss(recon_x, x, reduction='sum')  
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # return BCE + beta * KLD
        return MSE + beta * KLD



    def summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Latent dimension: {self.latent_dim}")


import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE2(nn.Module):
    def __init__(self, latent_dim=64, first_channel=32):
        super(VAE2, self).__init__()
        
        self.first_channel = first_channel
        self.latent_dim = latent_dim

        # Encoder: convolutional layers for grid-like data
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.first_channel, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.first_channel, 2*self.first_channel, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*self.first_channel, 4*self.first_channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*self.first_channel, 8*self.first_channel, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Linear layers to output mean and log variance for latent space
        self.fc1 = nn.Linear(self.first_channel * 6*6, self.latent_dim)  # Mean in latent space
        self.fc2 = nn.Linear(self.first_channel * 6*6, self.latent_dim)  # Log variance
        
        # Decoder: Linear layer followed by transposed convolutions to output a grid
        self.fc3 = nn.Linear(self.latent_dim, self.first_channel*6*6)  # To reshape latent space back to (8*scale, 7, 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*self.first_channel, 4*self.first_channel, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(4*self.first_channel, 2*self.first_channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(2*self.first_channel, self.first_channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(self.first_channel, 1, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True)
        )
    

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc1(x)           # Mean
        logvar = self.fc2(x)       # Log variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random noise
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        x = self.fc3(z)  # (B, latent_dim) -> (B, 8*6*6)
        x = x.view(x.size(0), 8*self.scale, 7, 7)  # Reshape to (B, 8, 7, 7)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)  # Get mean and log variance
        z = self.reparameterize(mu, logvar)  # Sample from the latent space
        reconstructed_x = self.decode(z)  # Decode back to original space
        return reconstructed_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        MSE = F.mse_loss(recon_x, x, reduction='sum')  
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # return BCE + beta * KLD
        return MSE + beta * KLD

