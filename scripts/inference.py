import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.dataset import SRDataset
from scripts.model import SimpleViTSR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ========== Settings ==========
upscale = 4
lr_path = "data/DIV2K_train_LR_bicubic/X4"
hr_path = "data/DIV2K_train_HR"
model_path = "model_weights.pth"  # Change if saved differently
num_samples_to_show = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Dataset ==========
dataset = SRDataset(lr_path, hr_path)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========== Load Model ==========
model = SimpleViTSR(upscale=upscale)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ========== Inference and Visualization ==========
os.makedirs("results", exist_ok=True)

for i, (lr, hr) in enumerate(loader):
    lr = lr.to(device)
    hr = hr.to(device)

    with torch.no_grad():
        sr = model(lr)

    # Convert to numpy for plotting
    lr_np = lr[0].cpu().permute(1, 2, 0).numpy()
    sr_np = sr[0].cpu().permute(1, 2, 0).numpy()
    hr_np = hr[0].cpu().permute(1, 2, 0).numpy()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(lr_np)
    axs[0].set_title("Low-Resolution Input")
    axs[0].axis('off')

    axs[1].imshow(sr_np)
    axs[1].set_title("Super-Resolved Output")
    axs[1].axis('off')

    axs[2].imshow(hr_np)
    axs[2].set_title("High-Resolution Ground Truth")
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(f"results/sample_{i}.png")
    plt.show()

    if i + 1 == num_samples_to_show:
        break
