import torch
from torch.optim import Adam,AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import logging
import os
from datetime import datetime
from tqdm import tqdm

from dataloader import ConvectionDiffusionDataset
from baseline import FNO_solver


def setup_logger(log_dir, log_name="train.log"):
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_path = os.path.join(log_dir, log_name)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger

def train(model, train_loader, test_loader, optimizer, scheduler, epochs, device, save_path, logger):
    best_test_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info("\nEpoch %d/%d | Learning Rate: %.2e", epoch + 1, epochs, current_lr)
        
        # pbar = tqdm(train_loader, desc="Training")
        for coeffs in train_loader:
            optimizer.zero_grad()
            
            # coeffs are the PDE parameters (kappa, vx, vy, f) from the dataloader
            # Shape: (B, 4, H, W)
            coeffs = coeffs.to(device)
            
            # Extract PDE coefficients 'a' (kappa, vx, vy)
            a = coeffs[:, 0:3, ...]
            
            # Generate a random right-hand side (rhs) for each sample in the batch
            # This is the key part of the training strategy
            rhs = torch.randn_like(coeffs[:, 0:1, ...])
            
            # Get the model's approximation of A_inv(rhs)
            # This is the direct output of the preconditioner module
            u_pred = model.train_forward(a, rhs)
            
            # --- Calculate Loss ---
            # We check how well u_pred satisfies A*u = rhs.
            # Using the preconditioned loss for training for better stability.

            loss = model.loss_fn(u_pred, coeffs, rhs)
            # --- Optional: Hybrid Loss (as discussed previously) ---
            # This can prevent the model from hiding errors in high frequencies.
            # To use it, uncomment the following lines.
            # loss_pde = model.loss_fn(u_pred, coeffs, rhs)
            # alpha = 0.1
            # loss = loss + alpha * loss_pde
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # pbar.set_postfix({'Train Loss': f'{loss.item():.4e}'})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for coeffs_test in test_loader:
                coeffs_test = coeffs_test.to(device)
                a_test = coeffs_test[:, 0:3, ...]
                rhs_test = torch.randn_like(coeffs_test[:, 0:1, ...])
                
                u_pred_test = model.train_forward(a_test, rhs_test)
                
                # Validate with the standard, unfiltered PDE loss for a true measure of performance

                val_loss_item = model.loss_fn(u_pred_test, coeffs_test, rhs_test)
                test_loss += val_loss_item.item()
        
        avg_test_loss = test_loss / len(test_loader)
        scheduler.step()
        
        logger.info(
            "-> Avg Train Loss (Precond): %.4e | Avg Val Loss (PDE): %.4e",
            avg_train_loss,
            avg_test_loss,
        )
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info("  -> New best model saved with test loss: %.4e", best_test_loss)

if __name__ == '__main__':
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_PATH = 'data_ill/train'
    TEST_PATH = 'data_ill/test'
    
    # Preconditioning mode
    USE_PRECOND_RES = True 
    MODEL_NAME = "fno_solver_precond" if USE_PRECOND_RES else "fno_solver"
    
    # Model params
    RESOLUTION = (128, 128)
    DX = 1.0 / RESOLUTION[0]
    FEATURES = 32
    WAVENUMBER = [16, 16, 32] 
    PADDING = 8
    
    # Training params
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    
    SCHEDULER_STEP_SIZE = EPOCHS // 20
    SCHEDULER_GAMMA = 0.8

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    exp_dir = os.path.join("models", f"{MODEL_NAME}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    logger = setup_logger(exp_dir)
    logger.info("Using device: %s | Mode: %s", DEVICE, MODEL_NAME)
    
    train_dataset = ConvectionDiffusionDataset(TRAIN_PATH, data_num=4000, device='cpu')
    test_dataset = ConvectionDiffusionDataset(TEST_PATH, data_num=400, device='cpu')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = FNO_solver(
        N=RESOLUTION,
        dx=DX,
        num_features=FEATURES,
        wavenumber=WAVENUMBER,
        padding=PADDING,
        device=DEVICE,
        use_precond_res=USE_PRECOND_RES
    ).to(DEVICE)
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    

    
    model_save_path = os.path.join(exp_dir, f"{MODEL_NAME}.pt")
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        EPOCHS,
        DEVICE,
        model_save_path,
        logger,
    )

    logger.info("\nTraining complete.")