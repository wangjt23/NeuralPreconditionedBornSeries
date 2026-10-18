# Resolve repository imports when this file is run directly.
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helmholtz.runtime import add_runtime_arguments, configure_runtime, configure_data
import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from helmholtz.utils import Logger, Timer, trainable_params_size, save_loss_history
from helmholtz.dataloader import unspvd_sos_Dataset
from helmholtz.models.model3d import MGNO_BS_3D


def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    # Config
    if args.finetune:
        train_cfg = cfg['finetune']
    else:
        train_cfg = cfg['train']
    model_cfg = cfg['model']
    device = configure_runtime(cfg, args)
    data_cfg = configure_data(cfg, model_cfg, args)
    save_dir = train_cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    if args.finetune:
        lprint = Logger(save_path=os.path.join(save_dir, "finetune.log"))
    else:
        lprint = Logger(save_path=os.path.join(save_dir, "train.log"))

    lprint(f"Using device: {device}")
    lprint(f"Configuration:\n{yaml.dump(cfg)}")

    # Build model
    model_type = model_cfg['name']
    if model_type == "mgno_bs_3d":
        model = MGNO_BS_3D(loss_type=model_cfg['loss_type'],
                           num_channel_u=model_cfg.get('num_channel_u', 32),
                           num_iteration=model_cfg.get('num_iteration', None),
                           activation=model_cfg.get('activation', 'gelu'),
                           freq=data_cfg.freq,
                           sos_mean=data_cfg.sos_mean,
                           sos_max=data_cfg.sos_max,
                           sos_min=data_cfg.sos_min,
                           N=data_cfg.size,
                           dx=data_cfg.dx,
                           boundary_width=model_cfg['boundary_width'],
                           boundary_type=model_cfg['boundary_type'],
                           fdtype=torch.float32,
                           device=device)
    else:
        raise ValueError(f"model '{model_type}' not supported")

    if args.finetune:
        orig_pth = os.path.join(save_dir, f"{model_type}_{args.load_weights_name}.pth")
        weights_suffix = "weights_finetune"
        lprint(f"Loading weights from {orig_pth}")
        state = torch.load(orig_pth, weights_only=True, map_location=device)
        model.load_state_dict(state)
        lprint("Pretrained weights loaded into model.")
    else:
        weights_suffix = "weights"

    lprint(f"Trainable params size: {trainable_params_size(model):.2f}MB")
    lprint("Start loading dataset...")
    # Datasets (expects 3D volumes (B,1,D,H,W))
    train_ds = unspvd_sos_Dataset(
        sos_path=data_cfg.dir_path,
        data_size=data_cfg.size,
        data_type=data_cfg.data_type,
        data_num=train_cfg['num_train'],
        test=False,
    )
    test_ds = unspvd_sos_Dataset(
        sos_path=data_cfg.dir_path,
        data_size=data_cfg.size,
        data_type=data_cfg.data_type,
        data_num=train_cfg['num_test'],
        test=True,
        test_point=train_cfg['val_point'],
    )

    batch_size = train_cfg['batch_size']
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    lprint(f"Dataset with size {data_cfg.size} loaded...")

    lr = float(train_cfg['lr'])
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg['lr_step_size'],
        gamma=train_cfg['lr_gamma'],
    )

    # Training loop
    num_epochs = train_cfg['num_epochs']
    log_steps = train_cfg['log_steps']
    val_steps = train_cfg['val_steps']
    train_loss_batch = []
    train_loss_epoch = []
    val_loss_step = []
    lprint('Start training...')
    timer = Timer(device=device)
    for epoch in range(num_epochs):
        model.train()
        t1 = timer.record()
        for batch_idx, sos in enumerate(train_loader, start=1):
            sos = sos.to(device)  # (B,1,D,H,W)
            model.setup(sos=sos)
            rhs = torch.randn(sos.size(0), 1, model.N_pad, model.N_pad, model.N_pad,
                              dtype=torch.complex64, device=device)
            if model_cfg['loss_type'] == 1:
                rhs_float = torch.cat([rhs.real, rhs.imag], dim=1)
                out = model.precond(rhs_float, model.a_list, model.s_list)
                out = torch.view_as_complex(out.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
                loss = torch.norm(rhs - model.A_op(out)) / torch.norm(rhs)
                loss_record = loss
            elif model_cfg['loss_type'] == 2:
                rhs_G = model.G(rhs / (model.dx ** 2))
                rhs_float = torch.cat([rhs_G.real, rhs_G.imag], dim=1)
                out = model.precond(rhs_float, model.a_list, model.s_list)
                out = torch.view_as_complex(out.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
                loss = torch.norm(rhs - model.A_op(out)) / torch.norm(rhs)
                loss_record = loss
            elif model_cfg['loss_type'] == 3:
                rhs_G = model.G(rhs / (model.dx ** 2))
                rhs_float = torch.cat([rhs_G.real, rhs_G.imag], dim=1)
                out = model.precond(rhs_float, model.a_list, model.s_list)
                out = torch.view_as_complex(out.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
                loss = torch.norm(rhs_G - model.G(model.A_op(out) / (model.dx ** 2))) / torch.norm(rhs_G)
                with torch.no_grad():
                    loss_record = torch.norm(rhs - model.A_op(out)) / torch.norm(rhs)
            else:
                raise NotImplementedError("ERROR!!")

            train_loss_batch.append(loss_record.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % log_steps == 0:
                lprint(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {train_loss_batch[-1]:.4e} "
                )
        t2 = timer.record()
        epoch_loss = sum(train_loss_batch[-len(train_loader):]) / len(train_loader)
        train_loss_epoch.append(epoch_loss)
        scheduler.step()
        epoch_print = f"Epoch [{epoch+1}/{num_epochs}] Avg Loss: {epoch_loss:.4e} Time used: {timer.elapsed_time(t1,t2):.2f}s Lr: {scheduler.get_last_lr()[0]:.2e}"
        lprint(epoch_print)
        lprint('-' * len(epoch_print))
        # Validation
        val_loss_batch = []
        if (epoch + 1) % val_steps == 0:
            model.eval()
            with torch.no_grad():
                for b_idx, sos in enumerate(test_loader, start=1):
                    sos = sos.to(device)
                    model.setup(sos=sos)
                    rhs = torch.randn(sos.size(0), 1, model.N_pad, model.N_pad, model.N_pad,
                                      dtype=torch.complex64, device=device)
                    if model_cfg['loss_type'] == 1:
                        rhs_float = torch.cat([rhs.real, rhs.imag], dim=1)
                        out = model.precond(rhs_float, model.a_list, model.s_list)
                        out = torch.view_as_complex(out.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
                        rel_error = torch.norm(rhs - model.A_op(out)) / torch.norm(rhs)
                        loss_record = rel_error
                    elif model_cfg['loss_type'] == 2:
                        rhs_G = model.G(rhs / (model.dx ** 2))
                        rhs_float = torch.cat([rhs_G.real, rhs_G.imag], dim=1)
                        out = model.precond(rhs_float, model.a_list, model.s_list)
                        out = torch.view_as_complex(out.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
                        rel_error = torch.norm(rhs - model.A_op(out)) / torch.norm(rhs)
                        loss_record = rel_error
                    elif model_cfg['loss_type'] == 3:
                        rhs_G = model.G(rhs / (model.dx ** 2))
                        rhs_float = torch.cat([rhs_G.real, rhs_G.imag], dim=1)
                        out = model.precond(rhs_float, model.a_list, model.s_list)
                        out = torch.view_as_complex(out.permute(0, 2, 3, 4, 1).unsqueeze(1).contiguous())
                        rel_error = torch.norm(rhs_G - model.G(model.A_op(out) / (model.dx ** 2))) / torch.norm(rhs_G)
                        loss_record = torch.norm(rhs - model.A_op(out)) / torch.norm(rhs)
                    else:
                        raise NotImplementedError("ERROR!!")
                    val_loss_batch.append(loss_record.item())
                    lprint(
                        f"Epoch [{epoch+1}/{num_epochs}] "
                        f"Val Batch [{b_idx}/{len(test_loader)}] "
                        f"Val Loss : {loss_record.item():.4e}"
                    )
                val_loss_step.append(sum(val_loss_batch[-len(test_loader):]) / len(test_loader))
                # save current best model
                pth_path = os.path.join(save_dir, f"{model_type}_{weights_suffix}.pth")
                torch.save(model.state_dict(), pth_path)
                val_print = f"Epoch [{epoch+1}/{num_epochs}] Avg val Loss: {val_loss_step[-1]:.4e}"
                lprint(val_print)
                lprint('-' * len(val_print))

            if args.finetune:
                save_loss_history(save_dir, train_loss_batch, filename="finetune_loss_batch", x_label="Batch")
                save_loss_history(save_dir, train_loss_epoch, filename="finetune_loss_epoch", x_label="Epoch")
                save_loss_history(save_dir, val_loss_step, filename="finetune_val_loss_step", x_label="Step")
            else:
                save_loss_history(save_dir, train_loss_batch, filename="train_loss_batch", x_label="Batch")
                save_loss_history(save_dir, train_loss_epoch, filename="train_loss_epoch", x_label="Epoch")
                save_loss_history(save_dir, val_loss_step, filename="val_loss_step", x_label="Step")

    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_type}_{weights_suffix}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MGNO_BS_3D")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--finetune", "-f",
        action="store_true",
        help="If set, load existing weights and finetune with a lower LR schedule"
    )
    parser.add_argument(
        "--load_weights_name", "-lwn",
        type=str,
        required=False,
        default="weights",
        help="pretrained weights name, weights or weights_finetune"
    )
    add_runtime_arguments(parser, inference=False)
    args = parser.parse_args()

    main(args)
