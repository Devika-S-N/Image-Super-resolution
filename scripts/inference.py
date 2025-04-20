from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from scripts.dataset import SRDataset
#from scripts.model import SimpleViTSR
from scripts.model_custom_swinir import SwinIR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

psnr_list, ssim_list = [], []
# ========== Settings ==========
upscale = 4
lr_path = "data/DIV2K_train_LR_bicubic/X4"
hr_path = "data/DIV2K_train_HR"
model_path = "model_weights_SWIN.pth"  # Change if saved differently
num_samples_to_show = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
assert os.path.exists(lr_path), f"LR path does not exist: {lr_path}"
assert os.path.exists(hr_path), f"HR path does not exist: {hr_path}"


# ========== Load Dataset ==========
dataset = SRDataset(lr_path, hr_path)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# ========== Load Model ==========
#model = SimpleViTSR(upscale=upscale)
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

    # Clip values to [0, 1] just in case
    sr_np = np.clip(sr_np, 0, 1)
    hr_np = np.clip(hr_np, 0, 1)

    # Compute PSNR & SSIM
    psnr_val = psnr(hr_np, sr_np, data_range=1.0)
    ssim_val = ssim(hr_np, sr_np, channel_axis=-1, data_range=1.0)
    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    print(f"Sample {i}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")


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
    #plt.show()

    if i + 1 == num_samples_to_show:
        break
print(f"\nAverage PSNR: {np.mean(psnr_list):.2f} dB")
print(f"Average SSIM: {np.mean(ssim_list):.4f}")