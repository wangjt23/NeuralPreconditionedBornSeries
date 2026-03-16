import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob

class ConvectionDiffusionDataset(Dataset):
    """
    PyTorch Dataset for loading the convection-diffusion data.
    
    This version loads the entire dataset into RAM during initialization 
    for maximum data loading speed during training, assuming sufficient
    system memory is available.
    """
    def __init__(self, data_path, data_num=5000,device='cpu'):
        """
        Args:
            data_path (str): Path to the directory containing the .pt files.
            device (str): The device to move the loaded data to ('cpu' or 'cuda').
        """
        super().__init__()
        self.data_path = data_path
        self.file_paths = sorted(glob.glob(os.path.join(data_path, "*.pt")))[:data_num]
        self.device = device
        
        if not self.file_paths:
            raise FileNotFoundError(f"No .pt files found in {data_path}. Please run the data generation script first.")

        self.samples = []
        print(f"Pre-loading and processing {len(self.file_paths)} samples into memory...")
        
        # Loop through all files and load them into a list in memory
        for file_path in (self.file_paths):
            sample = torch.load(file_path, map_location='cpu') # Load to CPU first
            
            # --- Prepare tensors ---
            kappa = sample['kappa']
            v = sample['v']         # Shape (H, W, 2)
            f = sample['f']
            
            # Create the sigma=1 tensor, as it's not saved in the files
            # sigma = torch.ones_like(kappa)
            
            # --- Ensure correct tensor shapes (C, H, W) ---
            # Add channel dimension if it's missing
            if kappa.ndim == 2: kappa = kappa.unsqueeze(0)
            # if sigma.ndim == 2: sigma = sigma.unsqueeze(0)
            if f.ndim == 2:     f = f.unsqueeze(0)
                
            # Permute velocity tensor from (H, W, C) to (C, H, W)
            v = v.permute(2, 0, 1) # Shape (2, H, W)
            
            # --- Stack all coefficient fields into a single tensor ---
            # The input will have 4 channels: kappa, vx, vy, f
            # Since you removed sigma from saving, we'll create it here.
            # Final channels: kappa, vx, vy, sigma, f (5 channels)
            # model_input = torch.cat([kappa, v, sigma, f], dim=0)
            model_input = torch.cat([kappa, v, f], dim=0)
            
            # Move to the target device and append to the list
            self.samples.append(model_input.to(self.device))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns the pre-loaded sample at the given index.
        This is very fast as the data is already in memory.
        """
        return self.samples[idx]

if __name__ == '__main__':
    # --- Example Usage ---
    
    # Define paths to the generated data
    train_data_path = 'data/train'
    test_data_path = 'data/test'
    
    # Use GPU for the dataloader if you plan to train on GPU.
    # This avoids CPU->GPU data transfer costs in the training loop.
    loader_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataloader will move data to: {loader_device}")


    train_dataset = ConvectionDiffusionDataset(train_data_path, data_num=2000,device=loader_device)
    test_dataset = ConvectionDiffusionDataset(test_data_path, data_num=200, device=loader_device)

    # Create DataLoader instances
    # For pre-loaded datasets, num_workers=0 is often fastest as there's no file I/O.
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\nNumber of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Fetch one batch to inspect its shape
    first_batch = next(iter(train_loader))
    
    print(f"\nShape of one batch of inputs: {first_batch.shape}")
    print(f"Device of the batch: {first_batch.device}")
    print(f"Expected shape: (batch_size, channels, height, width)")
    print(f"Current shape: ({batch_size}, 5, 128, 128)")
    
    print("\nDataloader is ready for training.")

