import time  
import torch
import os
from scripts.model_custom_swinir import SwinIR
from scripts.model import SimpleViTSR
from scripts.model_srgan import SRGAN_Generator
from scripts.dataset import SRDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from scripts.losses import VGGPerceptualLoss, SSIMLoss
import warnings
warnings.filterwarnings("ignore")

# Start time
start_time = time.time()

# Step 1: Load data - 50% training, 50% unused or for validation
lr_path = "data/DIV2K_train_LR_bicubic/X4"
hr_path = "data/DIV2K_train_HR"
dataset = SRDataset(lr_path, hr_path)
train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, test_data = random_split(dataset, [train_size, val_size])
loader = DataLoader(train_dataset, batch_size=4, shuffle=True)


# Step 2: Create model
model_selection = int(input("\n Select the model for training from the below options: \n 1. --> Basic Transformer Model \n 2. --> SWIN Transformer Model\n 3. --> CNN Model \n -->"))
num_epochs_input = int(input("\n Enter the number of training epochs to run: \n -->"))


if model_selection == 1:
    model = SimpleViTSR(upscale=4)
    checkpoint_path = "checkpoint_Basic.pth"

elif model_selection == 2:
    model = SwinIR(
    upscale=4,
    img_size=(48, 48),        # Match your cropped input image size
    patch_size=1,
    in_chans=3,
    embed_dim=60,             # Or change to 96 for stronger features
    depths=[2, 2, 2, 2],      # Can be [6, 6, 6, 6] for stronger SwinIR
    num_heads=[6, 6, 6, 6],
    window_size=8,
    mlp_ratio=2.0,
    upsampler='pixelshuffle',  # or 'pixelshuffledirect'
    img_range=1.0,
    resi_connection='1conv'
    )
    checkpoint_path = "checkpoint_SWIN.pth"

elif model_selection == 3:
    model = SRGAN_Generator(
    in_channels=3,
    num_res_blocks=16,
    upscale_factor=4
    )
    checkpoint_path = "checkpoint_CNN.pth"

else:
    print("\n Invalid selection")
    exit()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)


# Step 3: Define loss function and optimizer
l1_loss = nn.L1Loss().to(device)
perceptual_loss = VGGPerceptualLoss(weight=0.01).to(device)
ssim_loss = SSIMLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)  

total_epochs = 1000  # Total epochs you want to train
num_epochs_per_run = num_epochs_input
start_epoch = 0  # default starting epoch

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1  # resume training from the next epoch
    print(f"Resuming training from epoch {start_epoch}...")
else:
    print("No checkpoint found, starting fresh training.")


# Step 4: Train for a few epochs
for epoch in range(start_epoch, min(start_epoch + num_epochs_per_run, total_epochs)):
    model.train()
    running_loss = 0.0

    for i, (lr, hr) in enumerate(loader):
        lr = lr.to(device)
        hr = hr.to(device)
        
        # Forward pass
        sr = model(lr)
        
        # Compute loss (e.g., combined L1, perceptual, SSIM losses)
        loss_l1 = l1_loss(sr, hr)
        loss_perc = perceptual_loss(sr, hr)
        loss_ssim = ssim_loss(sr, hr)
        loss = loss_l1 + loss_perc + 0.3 * loss_ssim
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{total_epochs}], Batch [{i}], L1 Loss: {loss_l1.item():.4f}, Perc Loss: {loss_perc.item():.4f}, Total Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}] finished with avg loss: {running_loss / len(loader):.4f}")

    # Save the checkpoint after every epoch
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": running_loss / len(loader),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# Step 5: Save the weights file
if model_selection == 1:
    torch.save(model.state_dict(), "model_weights_Basic_Transformer.pth")
elif model_selection == 2:
    torch.save(model.state_dict(), "model_weights_SWIN.pth")
elif model_selection == 3:
    torch.save(model.state_dict(), "model_weights_SRGAN.pth")


# End time and execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")