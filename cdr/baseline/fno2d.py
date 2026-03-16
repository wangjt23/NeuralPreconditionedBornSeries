import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=4, out_channels=1):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 
        self.fc0 = nn.Linear(in_channels, self.width)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=1)
        x = self.fc0(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
    def get_grid(self, shape, device):
        batchsize, _, size_x, size_y = shape
        
        # Create a 1D tensor for x coordinates
        gridx_1d = torch.linspace(0, 1, size_x, dtype=torch.float, device=device)
        # Reshape to (1, size_x, 1) and expand to (batchsize, size_x, size_y)
        # This is a memory-efficient operation that reuses the data from the original tensor.
        gridx = gridx_1d.view(1, size_x, 1).expand(batchsize, size_x, size_y)
        
        # Create a 1D tensor for y coordinates
        gridy_1d = torch.linspace(0, 1, size_y, dtype=torch.float, device=device)
        # Reshape to (1, 1, size_y) and expand to (batchsize, size_x, size_y)
        gridy = gridy_1d.view(1, 1, size_y).expand(batchsize, size_x, size_y)
        return torch.stack([gridx, gridy], dim=1)



if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    # --- Model Parameters ---
    modes = 12       # Fourier modes
    width = 20       # Latent channel dimension
    nlayers = 4      # Number of Fourier layers
    
    # --- Input Parameters ---
    batch_size = 4
    height = 70
    width_spatial = 70
    
    print("--- Initializing FNO2d Model ---")
    model = FNO2d(modes1=modes, modes2=modes, width=width, nlayers=nlayers,padding=10).to(device)
    print("Model initialized successfully.")
    
    # --- Create a dummy input tensor ---
    # This simulates a batch of Speed-of-Sound (SOS) maps
    # Shape: (batch_size, channels, height, width)
    input_tensor = torch.randn(batch_size, 1, height, width_spatial).to(device)
    print(f"\nInput tensor shape: {input_tensor.shape}")
    
    # --- Perform a forward pass ---
    print("Performing forward pass...")

    output = model(input_tensor)
    print("Forward pass successful.")


    # --- Verify the output shape ---
    # The expected output should be (batch_size, 98, height, width)
    # where 98 is the number of time steps in the wavefield.
    expected_shape = (batch_size, 98, height, width_spatial)
    print(f"\nOutput tensor shape: {output.shape}")
    print(f"Expected output shape: {expected_shape}")
    
    assert output.shape == expected_shape, "Output shape is incorrect!"
    print("Test Passed: The output shape is correct.")