import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

class NetworkTrainer:
    """text"""
    def __init__(self, model, model_kwargs, device, learning_rate=0.01, beta_start=1.0):
        self.model = model(**model_kwargs)
        self.load_parameters()
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.lr_lambda = lambda epoch: max(0.0001, learning_rate * (1 - epoch / 500))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self.lr_lambda, last_epoch=-1)
        
        self.beta_current = beta_start # Beta annealing for KL divergence weight
        self.beta_decay_rate = 0.98  # Slower decay rate for smoother annealing

    def train_epoch(self, loader):
        
        running_loss = 0.0
        bce_loss = 0.0
        kld_loss = 0.0
        
        self.beta_current **= self.beta_decay_rate

        for batch in loader: 
            batch = batch.to(device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model.forward(batch)
            loss, bce, kld = self.model.loss_function(recon_batch, batch, mu, logvar, beta= 1/self.beta_current)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            bce_loss += bce
            kld_loss += kld

        return running_loss, bce_loss, kld_loss

    def train_network(self, data, num_epochs=50, print_every=10, save_every=100):

        self.model.train() 

        data_loader = DataLoader(data, batch_size=64, shuffle=True)
        data_loader_size = len(data_loader.dataset)

        start_time = time.time()

        for epoch in range(num_epochs):

            running_loss, bce_loss, kld_loss = self.train_epoch(data_loader)

            if (epoch + 1) % print_every == 0:
                self.print_progress(running_loss, bce_loss, kld_loss, data_loader_size, epoch, num_epochs, start_time)

            if (epoch + 1) % save_every == 0 or epoch == num_epochs:
                self.save_parameters()

        self.model.eval()

    def print_progress(self, loss, bce, kld, loader_size, epoch, num_epochs, start):
        
        epoch_loss = loss / loader_size
        bce = bce / loader_size
        kld = kld / loader_size
        total_time_sec = time.time() - start
        average_epoch_time = round(total_time_sec / (epoch + 1), 2)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: BCE+β*KLD = {bce:.1f} + β*{kld:.1f} = {epoch_loss:.2f} | Time {round(total_time_sec/60, 1):.2f} min at {average_epoch_time:.1f} sec/e") 
        

    def save_parameters(self, path=None):
        """Save model parameters to a file"""
        if path is None:
            # Create directory if it doesn't exist
            os.makedirs('model_parameters/vae', exist_ok=True)
            path = f'model_parameters/vae/{self.model.name}_parameters.pth'
        
        torch.save(self.model.state_dict(), path)
        print(f"Model parameters saved to {path}")


    def load_parameters(self, path=None):
        """Load model parameters from a file"""
        if path is None:
            path = f'model_parameters/vae/{self.model.name}_parameters.pth'
        
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path, weights_only=True))
            print(f"Model parameters loaded from {path}")
        else:
            print(f"No saved parameters found at {path}")

    def plot_images_and_reconstructions(self, data, num_subsamples=4, save_figure=False):
        
        data_loader = DataLoader(data, batch_size=128, shuffle=False)

        reconstructed_x = []
        with torch.no_grad(): # This performs a forward pass (without updating gradients)
            for batch in data_loader:
                mu, logvar = self.model.encode(batch)   
                recon_x    = self.model.decode(mu)     
                reconstructed_x.append(recon_x)
        
        reconstructed_x = torch.cat(reconstructed_x, dim=0)

        # Plotting original vs reconstructed versions
        indices = torch.randint(0, data.shape[0], (num_subsamples,))

        fig, axes = plt.subplots(2, num_subsamples, figsize=(2.2*num_subsamples, 6))
        for i, idx in enumerate(indices):
            axes[0, i].imshow(data.squeeze(1)[idx].cpu().numpy(), cmap='gray')
            axes[0, i].set_title(f'Input {idx.item()}', fontsize=8)
            axes[0, i].axis('off')  

            axes[1, i].imshow(reconstructed_x.squeeze(1)[idx].detach().numpy(), cmap='gray')
            axes[1, i].set_title(f'Recon {idx.item()}', fontsize=8)
            axes[1, i].axis('off')  
            
        plt.tight_layout()

        if save_figure:
            plt.savefig(f"./figures/original_and_reconstruction.png")
            plt.close(fig)

    def plot_distances_original_and_latent_space(self, data, save_figure=False):
        
        data_loader = DataLoader(data, batch_size=128, shuffle=False)

        reconstructed_x = []
        with torch.no_grad(): # This performs a forward pass (without updating gradients)
            for batch in data_loader:
                mu, logvar = self.model.encode(batch)   
                recon_x    = self.model.decode(mu)     
                reconstructed_x.append(recon_x)
        
        reconstructed_x = torch.cat(reconstructed_x, dim=0)

        distances_normal = torch.linalg.norm(inputs_train[:-1, 0, :, :] - inputs_train[1:, 0, :, :], dim=(1,2))
        distances_latent = torch.linalg.norm(reconstructed_x[:-1, 0, :, :] - reconstructed_x[1:, 0, :, :], dim=(1,2))

        fig, axes = plt.subplots(1, 2, figsize=(12,6))
        axes[0].scatter(range(10), distances_normal[:10])
        axes[0].set_title("Original representation")
        axes[1].scatter(range(10), distances_latent[:10])
        axes[1].set_title("Latent representation")

        if save_figure:
            plt.savefig(f"./figures/latent_distances_epoch.png")
            plt.close(fig)



class VAE_Heat(nn.Module):
    def __init__(self, latent_dim=64, first_channel=16, input_channels=1, input_size=(97, 97), max_channels=8):
        super(VAE_Heat, self).__init__()

        self.name = f"vae_upsample_ld{latent_dim}_fc{first_channel}_maxchannel{max_channels}"
        self.first_channel = first_channel
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.max_channels = max_channels
        self.input_size = input_size

        self.encoder = nn.Sequential(
            # Input: B x input_channels x 97 x 97
            nn.Conv2d(self.input_channels, self.first_channel, kernel_size=3, stride=2, padding=1),  # Output: 49 x 49
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.first_channel, 2 * self.first_channel, kernel_size=3, stride=2, padding=1), # Output: 25 x 25
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * self.first_channel, 4 * self.first_channel, kernel_size=3, stride=2, padding=1), # Output: 13 x 13
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * self.first_channel, self.max_channels, kernel_size=3, stride=2, padding=1), # Output: 7 x 7
            nn.LeakyReLU(0.2, inplace=True),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_channels, self.input_size[0], self.input_size[1])
            encoded_output = self.encoder(dummy_input)
            self.encoded_spatial_dim = encoded_output.shape[2] 

        self.flatten_size = self.max_channels * self.encoded_spatial_dim * self.encoded_spatial_dim

        self.fc1 = nn.Linear(self.flatten_size, self.latent_dim) # mu
        self.fc2 = nn.Linear(self.flatten_size, self.latent_dim) # logvar
        self.fc3 = nn.Linear(self.latent_dim, self.flatten_size) # latent to decoder input

        s = self.input_size[0]
        dims = []
        for _ in range(4): # Number of downsampling/upsampling stages
            s_out = ((s + 2*1 - 3) // 2) + 1
            dims.append(s_out)
            s = s_out
            
        target_dim1 = 13
        target_dim2 = 25
        target_dim3 = 49
        target_dim4 = self.input_size[0] # 97

        self.decoder = nn.Sequential(
            nn.Upsample(size=(target_dim1, target_dim1), mode='bilinear', align_corners=False),
            nn.Conv2d(self.max_channels, 4 * self.first_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(target_dim2, target_dim2), mode='bilinear', align_corners=False),
            nn.Conv2d(4 * self.first_channel, 2 * self.first_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(target_dim3, target_dim3), mode='bilinear', align_corners=False),
            nn.Conv2d(2 * self.first_channel, self.first_channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(size=(target_dim4, target_dim4), mode='bilinear', align_corners=False),
            nn.Conv2d(self.first_channel, self.input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Final activation
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc3(z)
        x = x.view(-1, self.max_channels, self.encoded_spatial_dim, self.encoded_spatial_dim)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        BCE = F.binary_cross_entropy(recon_x, x.view_as(recon_x), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + beta * KLD, BCE.item(), KLD.item()


