# Image Super-Resolution Project

This PyTorch-based repository implements image and video super-resolution using three distinct model architectures:

- **SwinIR Transformer**       (`scripts/model_custom_swinir.py`)
- **SimpleViTSR ViT model**    (`scripts/model.py`)
- **SRGAN CNN generator**      (`scripts/model_srgan.py`)

---

## 📂 Directory Structure

```
├── main.py                  # Entry point: training script
├── config.json              # JSON config for all modes
├── inference_final.py       # Inference/evaluation script
├── video_super_res.py       # Video decode/encode pipeline
├── download_div2k.py        # Download DIV2K dataset
├── scripts/                 # Python modules
│   ├── dataset.py           # SRDataset for LR/HR pairs
│   ├── modified_dataset.py  # SRDataset_modified for LR-only inference
│   ├── model_custom_swinir.py
│   ├── model.py             # SimpleViTSR definition
│   ├── model_srgan.py       # SRGAN_Generator & Discriminator
│   └── losses.py            # VGGPerceptualLoss & SSIMLoss
├── data/                    # Contains downloaded DIV2K zips & extracts
├── Video/                   # HR/LR video and frame folders
├── results_train/           # Outputs + logs from train-eval mode
├── results_test/            # Outputs from test mode
└── results_video/           # Outputs from video-mode
```

---

## ⚙️ Requirements

- Python 3.7+
- PyTorch & torchvision
- pytorch-msssim
- numpy, matplotlib, scikit-image, pillow
- tqdm, requests
- **FFmpeg** (for video scripts)

```bash
pip install torch torchvision pytorch-msssim numpy matplotlib scikit-image pillow tqdm requests
sudo apt-get update && sudo apt-get install -y ffmpeg
```

---

## 🔧 Configuration (`config.json`)

All configurable paths, hyperparameters, and model settings reside in **config.json**:

```json
{
  "device": "cuda",               
  "dirs": {                        
    "lr_train": "data/DIV2K_train_LR_bicubic/X4",
    "hr_train": "data/DIV2K_train_HR",
    "lr_test":  "Video/LR_Frames",
    "video_lr_train": "Video/LR_Frames",
    "video_hr_train": "Video/HR_Frames"
  },
  "model": {
    "architecture": "SwinIR",          
    "upscale": 4,                       
    "params": {                        
      "img_size": [48,48],             
      "patch_size": 1,                 
      "in_chans": 3,                   
      "embed_dim": 60,                 
      "depths": [2,2,2,2],             
      "num_heads": [6,6,6,6],          
      "window_size": 8,                
      "mlp_ratio": 2.0,                
      "upsampler": "pixelshuffle",   
      "img_range": 1.0,                
      "resi_connection": "1conv"     
    }
  },
  "train": {
    "batch_size": 4,
    "learning_rate": 1e-4,
    "total_epochs": 500,
    "epochs_per_run": 3,
    "loss_weights": {
      "l1": 1.0,
      "perceptual": 1.0,
      "ssim": 0.3
    }
  },
  "inference": {
    "results_dir_train": "results_train",
    "results_dir_test":  "results_test",
    "num_samples_test": 300,
    "num_samples_train":300
  }
}
```

### Tag Descriptions

- **device**: GPU/CPU selection (`"cuda"` or `"cpu"`).
- **dirs**: file paths for low-res & high-res images used in training, testing, and video frames.
- **model.architecture**: choose `"SwinIR"`, `"SimpleViTSR"`, or `"SRGAN"` (case-insensitive flag in scripts).
- **model.upscale**: integer upscale factor (e.g. 2, 4, 8).
- **model.params** *(Transformer-specific)*: only used when training SwinIR.
  - `img_size`: input patch size for transformer.
  - `patch_size`: patch embedding size.
  - `in_chans`: number of input channels (usually 3 for RGB).
  - `embed_dim`: embedding dimension per token.
  - `depths`: list of depths per Swin stage.
  - `num_heads`: heads per stage.
  - `window_size`: attention window.
  - `mlp_ratio`: MLP expansion ratio.
  - `upsampler`: reconstruction method.
  - `img_range`: pixel value scaling.
  - `resi_connection`: residual conv type (`1conv` or `3conv`).
- **train**: hyperparameters for training loop
  - `batch_size`: minibatch size.
  - `learning_rate`: optimizer LR.
  - `total_epochs`: total epochs to converge.
  - `epochs_per_run`: number of epochs per invocation (for checkpointing).
  - `loss_weights`: weights for L1, perceptual, and SSIM components.
- **inference**: output directories and max sample counts for evaluation.

---

## 📥 Data Preparation

1. **Download DIV2K**:
   ```bash
   python download_div2k.py
   ```
   This will fetch and unzip HR & LR ×4 images under `data/`.

2. **Organize custom data** (if any):
   - Place low-/high-res training images in your own folders and update `dirs.lr_train` & `dirs.hr_train`.

---

## 🚀 Training Each Model

All training runs save intermediate checkpoints (`checkpoint_*.pth`) and final weights (`model_weights_*.pth`).

| Model       | Flag             | Command Example                                    | Notes                                                                                         |
|-------------|------------------|----------------------------------------------------|-----------------------------------------------------------------------------------------------|
| SwinIR      | `--transformer`  | `python main.py --transformer --config config.json`| Uses `model.params` for architecture. Adjust depths/heads for larger models.                  |
| SimpleViTSR | `--basic`        | `python main.py --basic --config config.json`      | Default ViT: embed_dim=96, patch_size=4, num_blocks=8. Only `upscale` is configurable.        |
| SRGAN CNN   | `--cnn`          | `python main.py --cnn --config config.json`        | CNN-based SRGAN. Uses 16 residual blocks by default; upscale via config.                      |

- **Resume training**: if `checkpoint_*.pth` exists, training will resume at saved epoch.
- **Validation split**: by default, 50% of dataset used for training, rest discarded (modify code to incorporate explicit validation).
- **Logging**: batch losses printed every 50 iterations.

---

## 🔍 Inference & Evaluation

Generate super-resolved outputs and compute metrics:

```bash
python inference_final.py --test       --model swinir
python inference_final.py --train-eval --model basic
python inference_final.py --video-mode --model cnn
```

- **--test**: LR-only benchmarking (no HR needed).
- **--train-eval**: LR→SR vs. HR, outputs PSNR & SSIM.
- **--video-mode**: process frames extracted by `video_super_res.py`.

Outputs placed in directories defined under `inference` keys.

---

## 🎬 Video Super-Resolution Pipeline

1. **Decode** (extract LR/HR frames & downscale video):
   ```bash
   python video_super_res.py --decode
   ```
2. **Train / Inference** on frames
3. **Encode** (stitch result frames to video):
   ```bash
   python video_super_res.py --encode -r 30
   ```

Ensure `ffmpeg` is installed.

---

## 📑 Results & Visualization

- `results_train/`: grid images [LR, SR, HR] + console PSNR/SSIM.
- `results_test/`: grid images [LR, SR].
- `results_video/`: per-frame grids & output video files.

---

## 📄 License

Released under MIT License. Feel free to fork and extend!

