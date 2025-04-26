#!/usr/bin/env python3
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model_custom_swinir import SwinIR
from scripts.model import SimpleViTSR
from scripts.model_srgan import SRGAN_Generator
from scripts.modified_dataset import SRDataset_modified
from scripts.dataset import SRDataset


def load_cfg(path="config.json"):
    print(f" Loading config from {path}")
    with open(path) as f:
        cfg = json.load(f)
    print(" Config keys:", list(cfg.keys()))
    return cfg


def build_model(mc: dict, device, model_choice: str):
    print(" Building model with parameters:")
    for k, v in mc["params"].items():
        print(f"   {k}: {v}")

    model_key_map = {
        "swinir": "SWIN",
        "basic": "BASIC",
        "cnn": "CNN"
    }
    weights_key = f"model_path_{model_key_map[model_choice]}"
    weight_path = mc.get(weights_key)
    if not weight_path:
        raise ValueError(f"Model path for '{model_choice}' not found in config JSON.")

    print(f"\n>>> Selected model: {model_choice.upper()}")
    print(f">>> Loading weights from file: {weight_path}\n")

    if model_choice == "swinir":
        model = SwinIR(
            upscale=mc["upscale"],
            img_size=tuple(mc["params"]["img_size"]),
            patch_size=mc["params"]["patch_size"],
            in_chans=mc["params"]["in_chans"],
            embed_dim=mc["params"]["embed_dim"],
            depths=mc["params"]["depths"],
            num_heads=mc["params"]["num_heads"],
            window_size=mc["params"]["window_size"],
            mlp_ratio=mc["params"]["mlp_ratio"],
            upsampler=mc["params"]["upsampler"],
            img_range=mc["params"]["img_range"],
            resi_connection=mc["params"]["resi_connection"],
        ).to(device)

    elif model_choice == "basic":
        model = SimpleViTSR(upscale=mc.get("upscale", 4)).to(device)

    elif model_choice == "cnn":
        model = SRGAN_Generator(
            in_channels=3,
            num_res_blocks=16,
            upscale_factor=mc.get("upscale", 4)
        ).to(device)

    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    print(" Model ready")
    return model


def save_grid(path: Path, imgs, titles):
    fig, ax = plt.subplots(1, len(imgs), figsize=(6 * len(imgs), 4))
    if len(imgs) == 1:
        ax = [ax]
    for a, im, t in zip(ax, imgs, titles):
        a.imshow(im)
        a.set_title(t)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true")
    group.add_argument("--train-eval", action="store_true")
    group.add_argument("--video-mode", action="store_true")

    parser.add_argument("--model", type=str, required=True,
                        choices=["swinir", "basic", "cnn"],
                        help="Choose model: swinir, basic, or cnn")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(" Using device:", device)

    model = build_model(cfg["model"], device, args.model)
    root = Path(__file__).resolve().parent

    if args.test:
        mode = "TEST"
        lr_dir = cfg["dirs"]["lr_test"]
        out_dir = root / "results_test"
        n        = cfg["inference"]["num_samples_test"]
        loader = DataLoader(SRDataset_modified(lr_dir), batch_size=1, shuffle=False)

    elif args.train_eval:
        mode = "TRAIN-EVAL"
        lr_dir = cfg["dirs"]["lr_train"]
        hr_dir = cfg["dirs"]["hr_train"]
        out_dir = root / "results_train"
        n        = cfg["inference"]["num_samples_train"]
        loader = DataLoader(SRDataset(lr_dir, hr_dir), batch_size=1, shuffle=False)

    else:  # video-mode
        mode = "VIDEO"
        lr_dir = cfg["dirs"]["video_lr_train"]
        hr_dir = cfg["dirs"]["video_hr_train"]
        out_dir = root / "results_video"
        n        = cfg["inference"]["num_samples_train"]
        loader = DataLoader(SRDataset(lr_dir, hr_dir), batch_size=1, shuffle=False)

    print(f" Mode: {mode}")
    print(f" Loading data from: {lr_dir}" + (f", {hr_dir}" if 'hr_dir' in locals() else ""))
    print(f" Saving outputs to: {out_dir} (max {n} samples)")
    out_dir.mkdir(parents=True, exist_ok=True)

    psnrs, ssims = [], []

    for i, batch in enumerate(loader):
        if i >= n:
            break
        print(f"\n-- Sample {i} --")
        if args.test:
            lr = batch.to(device)
            print("  Running inference on LR image")
            with torch.no_grad():
                sr = model(lr)
            lr_np = lr[0].cpu().permute(1, 2, 0).numpy()
            sr_np = np.clip(sr[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
            save_grid(out_dir / f"sample_{i}.png", [lr_np, sr_np], ["LR", "SR"])

        else:
            lr, hr = batch
            lr, hr = lr.to(device), hr.to(device)
            print("  Running inference on LR image")
            with torch.no_grad():
                sr = model(lr)
            lr_np = lr[0].cpu().permute(1, 2, 0).numpy()
            sr_np = np.clip(sr[0].cpu().permute(1, 2, 0).numpy(), 0, 1)
            hr_np = np.clip(hr[0].cpu().permute(1, 2, 0).numpy(), 0, 1)

            print("  Computing metrics")
            p_val = psnr(hr_np, sr_np, data_range=1.0)
            s_val = ssim(hr_np, sr_np, channel_axis=-1, data_range=1.0)
            print(f"  PSNR = {p_val:.2f} dB, SSIM = {s_val:.4f}")
            psnrs.append(p_val)
            ssims.append(s_val)

            save_grid(out_dir / f"sample_{i}.png", [lr_np, sr_np, hr_np], ["LR", "SR", "HR"])

    if not args.test:
        print(f"\n Average PSNR: {np.mean(psnrs):.2f} dB")
        print(f" Average SSIM: {np.mean(ssims):.4f}")


if __name__ == "__main__":
    main()
