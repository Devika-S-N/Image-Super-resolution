
import torch
#from scripts.model import SimpleViTSR
from scripts.model_custom_swinir import SwinIR
from scripts.dataset import SRDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

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
model = model.to(device)

# Step 3: Define loss function and optimizer
criterion = nn.L1Loss()  # "How far off are we?"
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # "How do we get better?"

# Step 4: Train for a few epochs
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (lr, hr) in enumerate(loader):
        lr = lr.to(device)
        hr = hr.to(device)

        # Forward pass — make a guess
        sr = model(lr)

        # Compute loss — how wrong was the guess?
        print("SR shape:", sr.shape)
        print("HR shape:", hr.shape)
        loss = criterion(sr, hr)

        # Backward pass — figure out how to improve
        optimizer.zero_grad()
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item()

        # Print progress every few batches
        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}], Loss: {loss.item():.4f}")

    # Show average loss for this epoch
    print(f"Epoch [{epoch+1}] finished with avg loss: {running_loss / len(loader):.4f}")

#torch.save(model.state_dict(), "model_weights.pth")
torch.save(model.state_dict(), "model_weights_SWIN.pth")
