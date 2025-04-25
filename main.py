import time
import json
import torch
import os
import argparse
from scripts.model_custom_swinir import SwinIR
from scripts.model import SimpleViTSR
from scripts.model_srgan import SRGAN_Generator
from scripts.dataset import SRDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from scripts.losses import VGGPerceptualLoss, SSIMLoss
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Train a super-resolution model based on config.json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--transformer', action='store_true', help='Use SwinIR Transformer model')
    group.add_argument('--basic', action='store_true', help='Use SimpleViTSR basic transformer model')
    group.add_argument('--cnn', action='store_true', help='Use SRGAN CNN model')
    parser.add_argument('--config', type=str, default='config.json', help='Path to JSON configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device_str = config.get("device", "cuda")
    device = torch.device(device_str if device_str == "cuda" and torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dirs = config.get("dirs", {})
    lr_train = dirs.get("lr_train")
    hr_train = dirs.get("hr_train")
    dataset = SRDataset(lr_train, hr_train)
    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = random_split(dataset, [train_size, val_size])
    batch_size = config.get("train", {}).get("batch_size", 4)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if args.transformer:
        model_cfg = config.get("model", {})
        params    = model_cfg.get("params", {})
        model = SwinIR(
            upscale         = model_cfg.get("upscale", 4),
            img_size        = tuple(params.get("img_size", (48, 48))),     
            patch_size      = params.get("patch_size", 1),
            in_chans        = params.get("in_chans", 3),
            embed_dim       = params.get("embed_dim", 60),                   
            depths          = params.get("depths", [2, 2, 2, 2]),        
            num_heads       = params.get("num_heads", [6, 6, 6, 6]),
            window_size     = params.get("window_size", 8),
            mlp_ratio       = params.get("mlp_ratio", 2.0),
            upsampler       = params.get("upsampler", "pixelshuffle"),
            img_range       = params.get("img_range", 1.0),
            resi_connection = params.get("resi_connection", "1conv")
        )
        checkpoint_path = "checkpoint_swinir.pth"
        weights_path = "model_weights_swinir.pth"
    elif args.basic:
        model = SimpleViTSR(upscale=config.get("model", {}).get("upscale", 4))
        checkpoint_path = "checkpoint_basic.pth"
        weights_path = "model_weights_basic.pth"
    elif args.cnn:
        model = SRGAN_Generator(
            in_channels=3,
            num_res_blocks=16,
            upscale_factor=config.get("model", {}).get("upscale", 4)
        )
        checkpoint_path = "checkpoint_cnn.pth"
        weights_path = "model_weights_cnn.pth"
    else:
        raise ValueError("Please specify one of --transformer, --basic, or --cnn")

    model = model.to(device)
    loss_weights = config.get("train", {}).get("loss_weights", {})
    l1_w = loss_weights.get("l1", 1.0)
    perc_w = loss_weights.get("perceptual", 1.0)
    ssim_w = loss_weights.get("ssim", 0.3)
    l1_loss_fn = nn.L1Loss().to(device)
    perc_loss_fn = VGGPerceptualLoss(weight=perc_w).to(device)
    ssim_loss_fn = SSIMLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get("train", {}).get("learning_rate", 1e-4))

    total_epochs = config.get("train", {}).get("total_epochs", 1000)
    epochs_per_run = config.get("train", {}).get("epochs_per_run", total_epochs)
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", {}))
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("No checkpoint found, starting fresh training.")

    t0 = time.time()
    for epoch in range(start_epoch, min(start_epoch + epochs_per_run, total_epochs)):
        model.train()
        running_loss = 0.0
        for i, (lr_img, hr_img) in enumerate(loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr = model(lr_img)
            loss_l1 = l1_w * l1_loss_fn(sr, hr_img)
            loss_perc = perc_loss_fn(sr, hr_img)
            loss_ssim = ssim_w * ssim_loss_fn(sr, hr_img)
            loss = loss_l1 + loss_perc + loss_ssim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{total_epochs}] Batch [{i}] - L1: {loss_l1.item():.4f}, Perc: {loss_perc.item():.4f}, SSIM: {loss_ssim.item():.4f}, Total: {loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}] completed with avg loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")
    print(f"Total training time: {time.time() - t0:.2f} seconds")

if __name__ == '__main__':
    main()
