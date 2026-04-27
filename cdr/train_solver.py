import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloader import ConvectionDiffusionDataset
from baseline import FNO_solver
import logging


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

        for coeffs in train_loader:
            optimizer.zero_grad()

            coeffs = coeffs.to(device)
            a = coeffs[:, 0:3, ...]
            rhs = torch.randn_like(coeffs[:, 0:1, ...])

            u_pred = model.train_forward(a, rhs)
            loss = model.loss_fn(u_pred, coeffs, rhs)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for coeffs_test in test_loader:
                coeffs_test = coeffs_test.to(device)
                a_test = coeffs_test[:, 0:3, ...]
                rhs_test = torch.randn_like(coeffs_test[:, 0:1, ...])

                u_pred_test = model.train_forward(a_test, rhs_test)
                test_loss += model.loss_fn(u_pred_test, coeffs_test, rhs_test).item()

        avg_test_loss = test_loss / len(test_loader)
        scheduler.step()

        logger.info(
            "-> Avg Train Loss: %.4e | Avg Val Loss: %.4e",
            avg_train_loss,
            avg_test_loss,
        )

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info("  -> New best model saved with test loss: %.4e", best_test_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data_ill/train')
    parser.add_argument('--test_path', default='data_ill/test')
    parser.add_argument('--use_precond_res', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_precond_loss', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--resolution', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--features', type=int, default=25)
    parser.add_argument('--wavenumber', type=int, nargs='+', default=[32, 32, 32])
    parser.add_argument('--padding', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler_gamma', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESOLUTION = tuple(args.resolution)
    DX = 1.0 / RESOLUTION[0]
    MODEL_NAME = "fno_solver_precond" if args.use_precond_res else "fno_solver"
    MODEL_NAME = MODEL_NAME + ("_R_eta" if args.use_precond_loss else "_L2")
    SCHEDULER_STEP_SIZE = args.epochs // 20

    timestamp = datetime.now().strftime("%m_%d_%H_%M_%S")
    exp_dir = os.path.join("models", f"{MODEL_NAME}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    config = {
        'model_name': MODEL_NAME,
        'use_precond_res': args.use_precond_res,
        'use_precond_loss': args.use_precond_loss,
        'resolution': list(RESOLUTION),
        'dx': DX,
        'features': args.features,
        'wavenumber': args.wavenumber,
        'padding': args.padding,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'seed': args.seed,
    }
    with open(os.path.join(exp_dir, 'args.json'), 'w') as f:
        json.dump(config, f, indent=2)

    logger = setup_logger(exp_dir)
    logger.info("Using device: %s | Mode: %s", DEVICE, MODEL_NAME)

    train_dataset = ConvectionDiffusionDataset(args.train_path, data_num=4000, device='cpu')
    test_dataset = ConvectionDiffusionDataset(args.test_path, data_num=400, device='cpu')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = FNO_solver(
        N=RESOLUTION,
        dx=DX,
        num_features=args.features,
        wavenumber=args.wavenumber,
        padding=args.padding,
        device=DEVICE,
        use_precond_res=args.use_precond_res,
        use_precond_loss=args.use_precond_loss,
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=args.scheduler_gamma)

    model_save_path = os.path.join(exp_dir, f"{MODEL_NAME}.pt")
    train(model, train_loader, test_loader, optimizer, scheduler, args.epochs, DEVICE, model_save_path, logger)

    logger.info("\nTraining complete.")
