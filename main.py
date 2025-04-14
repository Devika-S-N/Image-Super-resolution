import time  # <-- added to track execution time
import torch
import os
from scripts.model_custom_swinir import SwinIR
from scripts.dataset import SRDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from scripts.losses import VGGPerceptualLoss, SSIMLoss

# Start time
start_time = time.time()

# Step 1: Load data
lr_path = "data/DIV2K_train_LR_bicubic/X4"
hr_path = "data/DIV2K_train_HR"
dataset = SRDataset(lr_path, hr_path)
#loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 70% training, 30% unused or for validation
train_size = int(0.5 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, _ = random_split(dataset, [train_size, val_size])

loader = DataLoader(train_dataset, batch_size=4, shuffle=True)



# Step 2: Create model
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
#model = SimpleViTSR(upscale=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)


# Step 3: Define loss function and optimizer
#criterion = nn.L1Loss()  # "How far off are we?"
l1_loss = nn.L1Loss().to(device)
perceptual_loss = VGGPerceptualLoss(weight=0.01).to(device)
ssim_loss = SSIMLoss().to(device)
#l1_loss = nn.L1Loss()
#perceptual_loss = VGGPerceptualLoss(weight=0.01)  # You can tune this weight
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # "How do we get better?"

checkpoint_path = "checkpoint.pth"
total_epochs = 500  # Total epochs you want to train
num_epochs_per_run = 150
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

#torch.save(model.state_dict(), "model_weights.pth")
torch.save(model.state_dict(), "model_weights_SWIN.pth")
# End time and execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")