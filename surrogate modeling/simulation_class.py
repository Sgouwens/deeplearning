import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

from IPython.display import HTML

from utils import print_model_information, make_simulations, generate_gaussian_peaks

class SimulationModel:

    """Given optimization """
    def __init__(self, inputs, outputs, model, equation, equation_parameter, batch_size=16, initial_lr=0.1, load_model_parameters=False):
        
        self.equation = equation
        self.equation_parameter = equation_parameter

        inputs_torch = inputs.clone().detach().unsqueeze(1).float()
        output_torch = outputs.clone().detach().unsqueeze(1).float()
        
        train_ratio = 0.8
        num_train = int(len(inputs) * train_ratio)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_inputs, self.train_output = inputs_torch[:num_train], output_torch[:num_train]
        self.test_inputs, self.test_output = inputs_torch[num_train:], output_torch[num_train:]

        # Optimization options
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=initial_lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, 
            start_factor=0.005,  
            end_factor=0.0001,  
            total_iters=150)
        self.batch_size = batch_size
        
        # Create datasets
        self.train_dataset = TensorDataset(self.train_inputs, self.train_output)
        self.test_dataset = TensorDataset(self.test_inputs, self.test_output)

        # Create data loaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

        self.training_errors = torch.tensor([])
        self.testing_errors = torch.tensor([])

        if load_model_parameters:
            self.load_model()


    def save_model(self):
        """"""
        torch.save(self.model.state_dict(), f'model_parameters/{self.model.model_name}.pth')


    def load_model(self):
        """If model parameters are saved under the corresponding model name, they are loaded."""

        if os.path.isfile(f'model_parameters/{self.model.model_name}.pth'):
            print("Loading parameters of the previously fit model.")
            self.model.load_state_dict(torch.load(f'model_parameters/{self.model.model_name}.pth',
                                                  weights_only=True))
            
        
    def train(self, num_epochs=100, save_model=False, print_every=10):
        """"""

        start_time = time.time()
        
        for epoch in range(num_epochs):

            start_time_epoch = time.time()
            running_loss_train = 0.0
            running_loss_test = 0.0

            for batch_idx, (inputs, output) in enumerate(self.train_loader):
                inputs, output = inputs.to(self.device), output.to(self.device)

                self.optimizer.zero_grad()
                outputs = inputs

                # Train model K steps
                # K = 1  # Adjust as needed
                # for _ in range(K):
                output_model = self.model(outputs)
                
                loss = self.criterion(output_model, output)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                running_loss_train += loss.item()
                
            
            self.model.eval()
            with torch.no_grad():
                for batch_idx, (inputs, output) in enumerate(self.test_loader):
                    inputs, output = inputs.to(self.device), output.to(self.device)
                    output_model = self.model(inputs)
                    running_loss_test += self.criterion(output_model, output).item()

            
            # Track training and testing errors
            avg_train_loss = running_loss_train / len(self.train_loader)
            avg_test_loss = running_loss_test / len(self.test_loader)

            self.training_errors = torch.cat((self.training_errors, torch.tensor([avg_train_loss])))
            self.testing_errors = torch.cat((self.testing_errors, torch.tensor([avg_test_loss])))

            # save errors.
            self.scheduler.step()

            if (epoch + 1) % print_every == 0:

                total_time_sec = time.time() - start_time
                average_epoch_time = round(total_time_sec / (epoch + 1), 2)
                
                print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {1e6*1.0*avg_train_loss:.4f} | Test Loss: {1e6*1.0*avg_test_loss:.4f} | Total time {round(total_time_sec/60, 1):.2f} min at {average_epoch_time:.1f} sec per epoch (error is *10e6)")
                
                if save_model:
                    # Saving the model parameters, error plots and csv file containing the error values.
                    self.save_model()
                    self.plot_errors(save_training_plot=True)

                    with open(f'model_parameters/error_tables/{self.model.model_name} error table.csv', mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(['Epoch', 'Training Error', 'Testing Error'])
                        
                        for e, (train_error, test_error) in enumerate(zip(self.training_errors, self.testing_errors)):
                            # Convert tensor values to Python float using .item() method
                            writer.writerow([e + 1, train_error.item(), test_error.item()])



    def plot_errors(self, skip_first=0, save_training_plot=False):
        """"""
        plt.figure(figsize=(7, 4))
        plt.plot(1e6*1.0*self.training_errors[skip_first:], label='Training Error', color='blue')
        plt.plot(1e6*1.0*self.testing_errors[skip_first:], label='Testing Error', color='red', linestyle='--')
        
        plt.title(f'Errors for {self.model.model_name}', fontsize=8)
        plt.xlabel('Epochs', fontsize=7)
        plt.ylabel('MSE * 10e6', fontsize=7)
        plt.legend(fontsize=6)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.ylim(0, 10)
        plt.tight_layout()

        if save_training_plot:
            plt.savefig(f'model_parameters/error_plots/{self.model.model_name} training testing.png', dpi=300)
            plt.close()
        else:
            plt.show()



    def animate_solution(self, save_animation=False):
        
        inputs_new, output_new = make_simulations(number=1, k=1, nt=100, as_tensor=True, equation=self.equation, equation_parameter=self.equation_parameter)

        # Start process from t=0
        N = inputs_new.shape[0]-1

        solution_original_nn = torch.empty((N, 100, 100))
        solution_original_nn[0] = output_new[0]  # Remove both batch and channel dimensions

        with torch.no_grad():
            for i in range(1, N):
                input_tensor = solution_original_nn[i-1].unsqueeze(0).unsqueeze(0)
                output = self.model(input_tensor)
                solution_original_nn[i] = output[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        def update(frame):
            ax1.clear()
            ax2.clear()
            
            # Original solution
            frame_data = output_new[frame].squeeze().cpu().numpy()
            ax1.imshow(frame_data, cmap='gray')
            ax1.set_title(f"Time step: {frame} - Torch")
            ax1.axis('off')

            # Neural network solution t|t=0
            frame_data_nn = solution_original_nn.unsqueeze(1)[frame].squeeze().cpu().numpy()
            ax2.imshow(frame_data_nn, cmap='gray')
            ax2.set_title(f"Time step: {frame} - Torch NN")
            ax2.axis('off')
            
            return ax1, ax2

        animation = FuncAnimation(fig, update, frames=N, interval=50, repeat=False)

        plt.close(fig)
        
        if save_animation:
            animation.save(f"model_animations/animation_{self.model.model_name}.gif")
        
        return HTML(animation.to_jshtml())     
    