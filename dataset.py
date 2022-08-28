import numpy as np
import os
import cv2
from torch.utils.data import Dataset


class CellDataset(Dataset):
  def __init__(self, image_dirs, mask_dirs, transform=None):
    self.image_dirs = image_dirs
    self.mask_dirs = mask_dirs
    self.transform = transform
    self.images = (os.listdir(image_dirs)).sort()
    self.masks = (os.listdir(mask_dirs)).sort()
    self.aug_image_dirs = os.path.join(self.image_dirs, "AUG")
    self.aug_mask_dirs = os.path.join(self.mask_dirs, "AUG")
    self.aug_images = []
    self.aug_masks = []

  def __len__(self):
    return len(self.images)

  def __apply__(self, images_to_generate):
    for index in range(0, images_to_generate):
      img_path = os.path.join(self.image_dirs, self.images[index % self.len()])
      mask_path = os.path.join(self.mask_dirs, self.masks[index % self.len()])
      image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
      ret, mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)

      if self.transform is not None:
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

      image_name = os.path.basename(self.images[index % self.len()])
      mask_name = os.path.basename(self.masks[index % self.len()])
      new_image_name = "%s_%s.png" %(image_name[:-4], index)
      new_mask_name = "%s_%s.png" %(mask_name[:-4], index)
      os.chdir(self.aug_image_dirs)  
      cv2.imwrite(new_image_name, image)
      os.chdir(self.aug_mask_dirs)   
      cv2.imwrite(new_mask_name, mask)
  
  def __read_augmented__(self):
    self.aug_images = (os.listdir(self.aug_image_dirs)).sort()
    self.aug_masks = (os.listdir(self.aug_mask_dirs)).sort()
      


