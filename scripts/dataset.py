import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, crop_size=96, upscale=4):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.upscale = upscale
        self.filenames = sorted(os.listdir(self.lr_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        hr_filename = self.filenames[idx].replace('x4', '')
        hr_path = os.path.join(self.hr_dir, hr_filename)

        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")

        # Get center crop of LR and corresponding HR
        lr_crop = TF.center_crop(lr_img, (self.crop_size, self.crop_size))
        hr_crop = TF.center_crop(hr_img, (self.crop_size * self.upscale, self.crop_size * self.upscale))

        # Convert to tensors
        lr_tensor = TF.to_tensor(lr_crop)
        hr_tensor = TF.to_tensor(hr_crop)

        return lr_tensor, hr_tensor
