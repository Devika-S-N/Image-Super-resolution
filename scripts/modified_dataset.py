import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class SRDataset_modified(Dataset):
    def __init__(self, lr_dir, crop_size=96, upscale=4):
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.upscale = upscale
        self.filenames = sorted(os.listdir(self.lr_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.filenames[idx])
        lr_img = Image.open(lr_path).convert("RGB")
 
        lr_crop = TF.center_crop(lr_img, (self.crop_size, self.crop_size))

        lr_tensor = TF.to_tensor(lr_crop)

        return lr_tensor
