import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class CellDataset(Dataset):
  def __init__(self, image_dirs, mask_dirs, transform=None):
    self.image_dirs = image_dirs
    self.mask_dirs = mask_dirs
    self.transform = transform
    self.images = os.listdir(image_dirs)
    self.masks = os.listdir(mask_dirs)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    img_path = os.path.join(self.image_dirs)
