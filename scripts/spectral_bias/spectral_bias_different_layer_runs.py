# %% [markdown]
# ### Experiments with training spectral bias

# %% [markdown]
# Imports

# %%
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import torch.optim as optim
import heapq
import os
from datetime import datetime
import argparse
import torch.nn.functional as F

# %% [markdown]
# Set save to True if you want to save plots

# %%
save = True

# %% [markdown]
# Only for python script: uncomment if running on cluster

# %%
# These are both placeholders
num_epochs = 1000
model_name = 'f0'

# %%
# # Parse command-line arguments
# parser = argparse.ArgumentParser(description='Train a neural network model')
# parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
# parser.add_argument('--model_name', type=str, default='model', help='Name of the saved model')
# args = parser.parse_args()

# # Use the parsed arguments
# num_epochs = args.epochs
# model_name = args.model_name

# %% [markdown]
# Check if CUDA is available

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
# For reproducibility
seed = 42

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed)

# %% [markdown]
# **Simplified function generation**

# %%
def generate_freq_dataset(num_samples, num_points, min_freq, max_freq):
    x = torch.linspace(0, 2 * np.pi, num_points, requires_grad=True)
    functions = []
    derivatives = []
    
    for _ in range(num_samples):
        # The number of different frequency components will be between 1 and 10
        num_freqs = torch.randint(1, 10, (1,)).item()
        amplitudes = torch.rand(num_freqs * 2)  # Double the number of amplitudes (between 0 and 1)
        frequencies = torch.randint(min_freq, max_freq + 1, (num_freqs,)).float()
        phases = torch.rand(num_freqs * 2) * 2 * np.pi  # Double the number of phases
        
        y = sum(a * torch.sin(f * x + p) for a, f, p in zip(amplitudes[:num_freqs], frequencies, phases[:num_freqs])) + \
            sum(a * torch.cos(f * x + p) for a, f, p in zip(amplitudes[num_freqs:], frequencies, phases[num_freqs:]))
        
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        
        functions.append(y.detach().numpy())
        derivatives.append(dy_dx.detach().numpy())
    
    return np.array(functions), np.array(derivatives)

# %% [markdown]
# ## Generate datasets

# %%
num_samples = 500
num_points = 1000
batch_size = 32

low_freq_functions, low_freq_derivatives = generate_freq_dataset(num_samples, num_points, 1, 5)
general_freq_functions, general_freq_derivatives = generate_freq_dataset(num_samples, num_points, 1, 10)
high_freq_functions, high_freq_derivatives = generate_freq_dataset(num_samples, num_points, 6, 10)

low_freq_dataset = TensorDataset(torch.tensor(low_freq_functions), torch.tensor(low_freq_derivatives))
general_freq_dataset = TensorDataset(torch.tensor(general_freq_functions), torch.tensor(general_freq_derivatives))
high_freq_dataset = TensorDataset(torch.tensor(high_freq_functions), torch.tensor(high_freq_derivatives))

low_freq_dataloader = DataLoader(TensorDataset(torch.tensor(low_freq_functions), torch.tensor(low_freq_derivatives)), batch_size=batch_size, shuffle=True)
high_freq_dataloader = DataLoader(TensorDataset(torch.tensor(high_freq_functions), torch.tensor(high_freq_derivatives)), batch_size=batch_size, shuffle=True)
general_freq_dataloader = DataLoader(TensorDataset(torch.tensor(general_freq_functions), torch.tensor(general_freq_derivatives)), batch_size=batch_size, shuffle=True)

# %%
total_size = len(low_freq_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
generator = torch.Generator().manual_seed(seed)

# low freq
train_dataset_l, test_dataset_l = random_split(low_freq_dataset, [train_size, test_size], generator=generator)
train_dataloader_l = DataLoader(train_dataset_l, batch_size=32, shuffle=True, drop_last=True)
test_dataloader_l = DataLoader(test_dataset_l, batch_size=32, shuffle=False, drop_last=True)

# general freq
train_dataset_g, test_dataset_g = random_split(general_freq_dataset, [train_size, test_size], generator=generator)
train_dataloader_g = DataLoader(train_dataset_g, batch_size=32, shuffle=True, drop_last=True)
test_dataloader_g = DataLoader(test_dataset_g, batch_size=32, shuffle=False, drop_last=True)

# high freq
train_dataset_h, test_dataset_h = random_split(high_freq_dataset, [train_size, test_size], generator=generator)
train_dataloader_h = DataLoader(train_dataset_h, batch_size=32, shuffle=True, drop_last=True)
test_dataloader_h = DataLoader(test_dataset_h, batch_size=32, shuffle=False, drop_last=True)


# %% [markdown]
# ## Plot random function from one of the datasets

# %%
a = np.random.randint(32)

fun, deriv = next(iter(low_freq_dataloader))

first_function = fun[a]
first_derivative = deriv[a]

# Generate x values corresponding to the function inputs
x_values = np.linspace(0, 2 * np.pi, 1000)

# Plotting the function and its derivative
plt.figure(figsize=(10, 6))
plt.plot(x_values, first_function, label='Function')
plt.plot(x_values, first_derivative, label='Derivative')
plt.title('Function and Its Derivative')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Miscellaneous plotting functions

# %%
def plot_losses(train_losses, test_losses, save_dir='plots', filename=None, save=False):
    if not train_losses:
        return
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    current_date = datetime.now().strftime("%m-%d")
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, linestyle='-', color='b', label='Training Loss')
    plt.plot(epochs, test_losses, linestyle='-', label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    plt.legend()
    plt.grid(True)
    filename = f"{filename}_{current_date}.png"
    save_path = os.path.join(save_dir, filename)
    if save:
        plt.savefig(save_path)    
    plt.show()

# %%
def plot_output(model1, dataset, order=None, save_dir='plots', filename=None, save=False): 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    current_date = datetime.now().strftime("%m-%d")
    model1.eval()  # Set the model to evaluation mode

    train_dataloader_viz = get_random_function(dataset, shuffle=True)
    # Get a random sample from the dataloader
    dataiter = iter(train_dataloader_viz)
    function, true_derivative = next(dataiter)

    # Reshape the input for the model
    function = function.unsqueeze(1)  # Add channel dimension

    # Make prediction
    with torch.no_grad():
        if order == 1 or order == 2:
            predicted_derivative = model1(function)
        
        if order == 'rollout':
            predicted_derivative = model1(function)
            predicted_derivative = model1(predicted_derivative)

    # Convert tensors to numpy arrays for plotting
    x = torch.linspace(0, 2*torch.pi, 1000).numpy()
    function = function.squeeze().numpy()

    predicted_derivative = predicted_derivative.squeeze().numpy()

    true_derivative = true_derivative.squeeze().numpy()

    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.plot(x, function, label='Original Function', color='blue')
    if order == 1:
        plt.plot(x, true_derivative, label=f'True {order}st derivative')

    plt.plot(x[10:-10], predicted_derivative[10:-10], label=f'Predicted {order}nd Derivative', linestyle='--')

    plt.title('Function, True Derivatives, and Predicted Derivatives')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    filename = f"{filename}_{current_date}.png"

    save_path = os.path.join(save_dir, filename)
    if save:
        plt.savefig(save_path)  
    plt.show()

# %%
def get_random_function(dataset, shuffle=True):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle)

# %% [markdown]
# ## Function to train model

# %%
criterion = nn.MSELoss()

# %%
train_losses, test_losses = [], []

def model_training(model, train_dataloader, test_dataloader, num_epochs,\
    split_freq=None, filename=None, save=None, order=None):
    train_losses = []
    test_losses = []

    low_freq_nmses = []
    general_freq_nmses = []
    high_freq_nmses = []
    epoch_list = []

    lr = 1e-3

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        test_loss = 0.0

        if epoch % split_freq == 0:
            l, g, h = print_and_store_metrics(model)
            low_freq_nmses.append(l)
            general_freq_nmses.append(g)
            high_freq_nmses.append(h)
            epoch_list.append(epoch)

        for batch_functions, batch_derivatives in train_dataloader:
            batch_functions = batch_functions.unsqueeze(1)
            batch_derivatives = batch_derivatives.unsqueeze(1)

            outputs = model(batch_functions)
            loss = criterion(outputs, batch_derivatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for b_test_functions, b_test_derivatives in test_dataloader:
                b_test_functions = b_test_functions.unsqueeze(1)
                b_test_derivatives = b_test_derivatives.unsqueeze(1)

                test_outputs = model(b_test_functions)
                batch_test_loss = criterion(test_outputs, b_test_derivatives)

                test_loss += batch_test_loss.item()

        test_loss /= len(test_dataloader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    print(f"Training finished for {order}st derivative")

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, low_freq_nmses, label='Low freq NMSE')
    plt.plot(epoch_list, general_freq_nmses, label='General freq NMSE')
    plt.plot(epoch_list, high_freq_nmses, label='High freq NMSE')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.yscale('log')
    plt.title('NMSEs of different frequencies during training')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    if save:
        plt.savefig(filename)  

    return train_losses, test_losses

# %% [markdown]
# ## Metric functions

# %%
def compute_mse(dataloader, model):
    """
    Takes in a dataloader and a model to compute MSE.
    """

    model.eval()
    all_outputs = []
    all_targets = []

    for function, deriv in dataloader:
        function = function.unsqueeze(1)
        deriv = deriv.unsqueeze(1)

        # Compute model output
        model_output = model(function)
        all_targets.append(deriv)

        # Collect outputs
        all_outputs.append(model_output)

    # Concatenate all collected outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute MSE
    mse = torch.mean((all_targets - all_outputs) ** 2)
    nmse = mse / torch.mean(all_targets ** 2)

    return mse.item(), nmse.item()

criterion = torch.nn.MSELoss()

# %%
def print_and_store_metrics(f0):
    return compute_mse(test_dataloader_l, f0)[1], compute_mse(test_dataloader_g, f0)[1], compute_mse(test_dataloader_h, f0)[1]

# %%
def print_metrics(model):
    print(f"NMSE over low freq test functions: {compute_mse(test_dataloader_l, model)[1]}")
    print(f"NMSE over general freq test functions: {compute_mse(test_dataloader_g, model)[1]}")
    print(f"NMSE over high freq test functions: {compute_mse(test_dataloader_h, model)[1]}")



# %% [markdown]
# ## Create models

# %%
# simple 1D CNN
class SimpleCNN(nn.Module):
    def __init__(self, n_layers=3, kernel_size=3, hidden_size=64):
        super(SimpleCNN, self).__init__()
        # Parameters
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size

        # Model
        self.convs = nn.ModuleList()
        if n_layers == 1:
            self.convs.append(nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2))
        elif n_layers >= 2:
            self.convs.append(nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2))
            for _ in range(n_layers - 2):
                self.convs.append(nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2))
            self.convs.append(nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=kernel_size//2))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                x = self.relu(conv(x))
            else:
                x = conv(x)
        return x

# %%
set_seed(seed)
L3ModelK3 = SimpleCNN(n_layers=3, kernel_size=3)

# %%
# train_losses, test_losses = model_training(L3ModelK3, train_dataloader_g, test_dataloader_g,\
#     num_epochs=1000, split_freq=10, filename='../plots/spectral_bias/f0_E2000_phased_training',\
#     save=save, order='first')
# plot_losses(train_losses, test_losses, save_dir='../plots/spectral_bias', filename='f0_E2000_phased_training', save=save)

# %% [markdown]
# ## Plot results

# %%
def plot_all_outputs(model, model_name):
    plot_output(model, dataset=test_dataset_l, order=1, save_dir='../plots/spectral_bias', filename=f'{model_name}_E{num_epochs}_lf_output', save=save)
    plot_output(model, dataset=test_dataset_g, order=1, save_dir='../plots/spectral_bias', filename=f'{model_name}_E{num_epochs}_lf_output', save=save)
    plot_output(model, dataset=test_dataset_h, order=1, save_dir='../plots/spectral_bias', filename=f'{model_name}_E{num_epochs}_lf_output', save=save)

# %%
plot_all_outputs(L3ModelK3, model_name = 'L3ModelK3')

# %%
print_metrics(L3ModelK3)

# %%
print(L3ModelK3)

# %% [markdown]
# ## Color map plots

# %%
def compute_fft_and_max_freq(dataloader, deriv=False, model=None, residue=False):
    fft_amplitudes = []
    max_frequencies = []
    T = 2 * torch.pi
    N = 1000

    # The spacing is T / N, i.e. 2pi/1000, but since we interpret f(x)=sin(5x) to
    # have a frequency of 5 over the domain x=[0,2pi], then we scale up by 2pi 
    # to get the unit cycle back to 1
    frequencies = torch.fft.fftfreq(N, T / N) * T
    positive_freq_indices = frequencies >= 0
    positive_freqs = frequencies[positive_freq_indices]

    plot_type = ''

    # Iterate over each batch
    for functions, derivatives in dataloader:  # Note that derivatives are ignored in this loop
        
        if deriv and not model and not residue: # If you only want the derivative
            functions = derivatives
            F = torch.fft.fft(functions)
            plot_type = "ground truth u_g'"
            
            
        elif not deriv and model and not residue: # If you only want the model output
            functions = model(functions.unsqueeze(1)).squeeze()
            F = torch.fft.fft(functions)
            plot_type = 'model output f(u_g)'

            # output = model(functions)
            
            # output = output.squeeze()
            # functions = output # set this so that the FFTs can be computed in the next line
        
        elif residue and model: # If you only want the residue
            
            functions = functions.unsqueeze(1)
            outputs = model(functions).squeeze()
            F_outputs = torch.fft.fft(outputs)

            F_derivatives = torch.fft.fft(derivatives)

            residues = F_derivatives - F_outputs
            
            # print(f"shape of F_derivaives: {F_derivatives.shape}")
            # normalizing = F_derivatives.norm(p=2, dim=1, keepdim=True) ** 2 / F_derivatives.shape[1]
            # print(f"shape of normalizing: {normalizing}")
            # residues = residues / normalizing

            plot_type = 'spectral error'
            F = residues
            # print(f"F is: {F}")

        else:
            plot_type = 'ground truth u_g'
            F = torch.fft.fft(functions)

        # else: # If you only want the original function u_g
        magnitudes = torch.abs(F) / N

        # Consider only positive frequencies
        positive_magnitudes = magnitudes[:, positive_freq_indices]

        fft_amplitudes.append(positive_magnitudes)
        
        # Maximum frequency based on the highest amplitude for each function in the batch
        max_indices = torch.argmax(positive_magnitudes, dim=1)
        batch_max_freqs = positive_freqs[max_indices]
        max_frequencies.extend(batch_max_freqs)
        print(f"Plotting {plot_type}")

    return torch.vstack(fft_amplitudes), torch.tensor(max_frequencies), positive_freqs, plot_type

def plot_heatmap(fft_amplitudes, max_frequencies, freqs, fun_type, xmin=0, xmax=0,\
    first=False, sorted_indices=None):

    fft_amplitudes = fft_amplitudes.detach().numpy()
    max_frequencies = max_frequencies.detach().numpy()
    freqs = freqs.detach().numpy()

    
    if first:
    # Sort functions by dominant frequency
        sorted_indices = np.argsort(-max_frequencies)  # Sort in descending order
        sorted_fft = fft_amplitudes[sorted_indices]
    else:
        print("Using predefined sort")
        sorted_fft = fft_amplitudes[sorted_indices]

    plt.figure(figsize=(10, 6))
    im = plt.imshow(sorted_fft, aspect='auto', extent=[freqs[0], freqs[-1], 0, len(sorted_fft)],\
        interpolation='nearest')

    plt.colorbar(im, label='Amplitude')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Function Index (sorted by max frequency)')
    plt.title(f'FFT Amplitude Heatmap for {fun_type}')
    plt.xlim([xmin, xmax])
    plt.show()

    if first:
        return sorted_indices


# %% [markdown]
# ## Training function

# %%
train_losses, test_losses = [], []

def model_training_plots(model, train_dataloader, test_dataloader, num_epochs,\
    split_freq=None, filename=None, save=None, order=None, nmse=False,\
        deriv=None, residue=False, lr=1e-3, heatmap=True):
    train_losses = []
    test_losses = []

    low_freq_nmses = []
    general_freq_nmses = []
    high_freq_nmses = []
    epoch_list = []

    lr = lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()
    # If nmse, then use NMSE as loss
    if nmse:
        def criterion(target, output, nmse=None):
            mse = torch.mean((target - output) ** 2)
            mse = mse / torch.mean(target ** 2)
            
            return mse
    
    num_plots = split_freq
    split_freq = num_epochs // split_freq
    print(split_freq)

    # At the first epoch, compute the order of the functions before training
    sorted_indices = plot_heatmaps(label=f'first', all=False, first=True)
    plt.show()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        test_loss = 0.0

        for batch_functions, batch_derivatives in train_dataloader:
            batch_functions = batch_functions.unsqueeze(1)
            batch_derivatives = batch_derivatives.unsqueeze(1)

            outputs = model(batch_functions)
            loss = criterion(outputs, batch_derivatives)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for b_test_functions, b_test_derivatives in test_dataloader:
                b_test_functions = b_test_functions.unsqueeze(1)
                b_test_derivatives = b_test_derivatives.unsqueeze(1)

                test_outputs = model(b_test_functions)
                batch_test_loss = criterion(test_outputs, b_test_derivatives)
                

                test_loss += batch_test_loss.item()

        test_loss /= len(test_dataloader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # If iteration reached, then plot the colormap once
        if heatmap:
            if (epoch) % split_freq == 0:
                print(f"Plotting the colormap once at iteration {epoch}")
                label = epoch // split_freq

                l, g, h = print_and_store_metrics(model)
                low_freq_nmses.append(l)
                general_freq_nmses.append(g)
                high_freq_nmses.append(h)
                epoch_list.append(epoch)

                # First is false here, but we pass in
                plot_heatmaps(model=model, label=f'{label}', all=False,\
                    deriv=deriv, residue=residue, epoch=epoch, sorted_indices=sorted_indices)

                plt.show()

    print(f"Training finished for {order}st derivative")

    plt.figure(figsize=(10, 6))
    plt.plot(epoch_list, low_freq_nmses, label='Low freq NMSE')
    plt.plot(epoch_list, general_freq_nmses, label='General freq NMSE')
    plt.plot(epoch_list, high_freq_nmses, label='High freq NMSE')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.yscale('log')

    plt.title('NMSEs of different frequencies during training')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    if save:
        plt.savefig(filename)  

    return train_losses, test_losses

# %% [markdown]
# ## Changing model depth

# %%
def train_varying_layers(layer_list, train_dataloader, test_dataloader, num_epochs,\
    seed, split_freq=None, filename=None, save=None, save_model=False, order=None):
    
    set_seed(seed)
    
    all_train_losses = []
    all_test_losses = []

    general_freq_nmse_dict = {}
    low_freq_nmse_dict = {}
    high_freq_nmse_dict = {}

    epoch_list = []

    avg_gen_nmse_last_50_epochs = np.zeros(len(layer_list))
    avg_low_nmse_last_50_epochs = np.zeros(len(layer_list))
    avg_high_nmse_last_50_epochs = np.zeros(len(layer_list))

    lr = 1e-3

    for i, layer in enumerate(layer_list):
        # Set the seed each time we increase the kernel size in the model
        set_seed(seed)
        model = SimpleCNN(n_layers=layer, kernel_size=3)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        train_losses = []
        test_losses = []
        general_freq_nmses = []
        low_freq_nmses = []
        high_freq_nmses = []

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            test_loss = 0.0

            # Calculate the metrics at every iteration
            l, g, h = print_and_store_metrics(model)
            general_freq_nmses.append(g)
            low_freq_nmses.append(l)
            high_freq_nmses.append(h)

            epoch_list.append(epoch)

            for batch_functions, batch_derivatives in train_dataloader:
                batch_functions = batch_functions.unsqueeze(1)
                batch_derivatives = batch_derivatives.unsqueeze(1)

                outputs = model(batch_functions)
                loss = criterion(outputs, batch_derivatives)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dataloader)

            model.eval()
            with torch.no_grad():
                for b_test_functions, b_test_derivatives in test_dataloader:
                    b_test_functions = b_test_functions.unsqueeze(1)
                    b_test_derivatives = b_test_derivatives.unsqueeze(1)

                    test_outputs = model(b_test_functions)
                    batch_test_loss = criterion(test_outputs, b_test_derivatives)

                    test_loss += batch_test_loss.item()

            test_loss /= len(test_dataloader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        print(f"Training finished for {order}st derivative")

        # Store the losses for this model size
        all_train_losses.append(train_losses)
        all_test_losses.append(test_losses)

        # Store the general frequency NMSEs in the dictionary
        general_freq_nmse_dict[layer] = general_freq_nmses
        low_freq_nmse_dict[layer] = low_freq_nmses
        high_freq_nmse_dict[layer] = high_freq_nmses

        # Calculate and store the average NMSE over the last 50 epochs
        # for each of the freq dataseets
        avg_gen_nmse_last_50_epochs[i] = np.mean(general_freq_nmses[-50:])
        avg_low_nmse_last_50_epochs[i] = np.mean(low_freq_nmses[-50:])
        avg_high_nmse_last_50_epochs[i] = np.mean(high_freq_nmses[-50:])


    return all_train_losses, all_test_losses, general_freq_nmse_dict, low_freq_nmse_dict, high_freq_nmse_dict


# %%
def run_with_multiple_seeds2(layer_list, train_dataloader, test_dataloader, num_epochs, seeds, split_freq=None, filename=None, save=None, save_model=False, order=None):
    # Dictionary to store results for plotting the NMSE over all epochs
    all_epochs_nmse = {k: [] for k in layer_list}
    all_epochs_low_nmse = {k: [] for k in layer_list}
    all_epochs_high_nmse = {k: [] for k in layer_list}

    # Lists to store the averaged NMSEs over the last 50 epochs for error plotting
    all_avg_gen_nmse_last_50_epochs = []
    all_avg_low_nmse_last_50_epochs = []
    all_avg_high_nmse_last_50_epochs = []

    num_runs = len(seeds)

    for seed in seeds:
        print(f"Training with seed: {seed}")
        _, _, general_freq_nmse_dict, low_freq_nmse_dict, high_freq_nmse_dict = train_varying_layers(layer_list, train_dataloader, test_dataloader, num_epochs, seed, split_freq, filename, save, save_model, order)
        
        # Collect the detailed NMSE over all epochs for each kernel size
        for k in layer_list:
            all_epochs_nmse[k].append(general_freq_nmse_dict[k])
            all_epochs_low_nmse[k].append(low_freq_nmse_dict[k])
            all_epochs_high_nmse[k].append(high_freq_nmse_dict[k])
        
        # Collect the average NMSE over the last 50 epochs for each kernel size
        all_avg_gen_nmse_last_50_epochs.append([np.mean(general_freq_nmse_dict[k][-50:]) for k in layer_list])
        all_avg_low_nmse_last_50_epochs.append([np.mean(low_freq_nmse_dict[k][-50:]) for k in layer_list])
        all_avg_high_nmse_last_50_epochs.append([np.mean(high_freq_nmse_dict[k][-50:]) for k in layer_list])

    # Plotting NMSE over all epochs for each kernel size with mean and std deviation
    plt.figure(figsize=(20, 15))
    plt.subplot(2, 2, 1)

    # To show the first and last elements in the kernels dictionary
    first_key = next(iter(all_epochs_nmse))  # Get the first key
    last_key = next(reversed(all_epochs_nmse))  # Get the last key
    
    for l, values in all_epochs_nmse.items():
        if l == first_key or l == last_key:
            values = np.array(values)  # Convert list of lists to 2D numpy array
            mean_values = np.mean(values, axis=0)
            std_error = np.std(values, axis=0) / np.sqrt(num_runs)
            epochs = range(1, num_epochs + 1)
            plt.plot(epochs, mean_values, label=f'{l} Layers')
            plt.fill_between(epochs, mean_values - std_error, mean_values + std_error, alpha=0.2)


    plt.xlabel('Epoch')
    plt.ylabel('NMSE')
    plt.yscale('log')
    plt.title('Training NMSEs')
    plt.legend()

    # Calculate the mean and standard deviation of the NMSE across all runs for the last 50 epochs
    mean_gen = np.mean(all_avg_gen_nmse_last_50_epochs, axis=0)
    std_gen = np.std(all_avg_gen_nmse_last_50_epochs, axis=0) / np.sqrt(num_runs)
    mean_low = np.mean(all_avg_low_nmse_last_50_epochs, axis=0)
    std_low = np.std(all_avg_low_nmse_last_50_epochs, axis=0) / np.sqrt(num_runs)
    mean_high = np.mean(all_avg_high_nmse_last_50_epochs, axis=0)
    std_high = np.std(all_avg_high_nmse_last_50_epochs, axis=0) / np.sqrt(num_runs)

    # Plotting the average NMSE over the last 50 epochs with error bars for all datasets
    plt.subplot(2, 2, 2)
    plt.errorbar(layer_list, mean_low, yerr=std_low, fmt='s-', label='Mean Low Frequency NMSE', capsize=5)
    plt.errorbar(layer_list, mean_gen, yerr=std_gen, fmt='o-', label='Mean General Frequency NMSE', capsize=5)
    plt.errorbar(layer_list, mean_high, yerr=std_high, fmt='^-', label='Mean High Frequency NMSE', capsize=5)
    plt.xlabel('Layers')
    plt.ylabel('NMSE')
    plt.title('Mean NMSEs for Different Layers')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig(f'plots/{filename}_nmse_comparison.png')
    plt.show()

    return all_epochs_nmse, mean_gen, std_gen, mean_low, std_low, mean_high, std_high

# %%
layer_list = [3, 7, 11, 15, 19, 23, 27, 31, 35]
seeds = [1, 2, 3, 4, 5]

# %%
# layer_list = [3, 7, 11]
# seeds = [1, 2, 3]

results = run_with_multiple_seeds2(layer_list, train_dataloader_g, test_dataloader_g,\
    num_epochs=1000, seeds=seeds, split_freq=2,\
    filename="/home/users/erikwang/multistage/plots/spectral_bias/layers_5_runs_final",\
    save=True, save_model=False, order=None)

# %%



