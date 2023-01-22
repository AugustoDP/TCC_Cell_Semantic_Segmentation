import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import boundary_label_2d, binary_label

# TODO: Think about making this more generic and flexible, for now architecture is very
# stuck, it only works with a certain format of inputs, that's a bad smell
class CellDataset(Dataset):
  def __init__(self,
   images_dir,
   masks_dir,
   size,
   transform=None, 
   classes=None, 
   preprocessing=None
   ):
    self.image_ids = os.listdir(images_dir)
    self.mask_ids = os.listdir(masks_dir)
    self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
    self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mask_ids]
    self.images_dir = images_dir
    self.masks_dir = masks_dir
    self.transform = transform
    self.preprocessing = preprocessing
    self.classes = classes
    self.max_size = size

    self.images_fps.sort()
    self.masks_fps.sort()
    # convert str names to class values on masks
    self.class_values = [self.classes.index(cls) for cls in classes]
  def __len__(self):
    return len(self.image_ids)
    
  def __getitem__(self, i):
          
          # read data
          image = cv2.imread(self.images_fps[i])
          image = cv2.resize(image, (self.max_size, self.max_size))
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_UNCHANGED)
          mask = cv2.resize(mask, (self.max_size, self.max_size))

          
          # apply preprocessing
          if self.preprocessing:
              sample = self.preprocessing(image=image, mask=mask)
              image, mask = sample['image'], sample['mask']
          label_boundary = boundary_label_2d(label=mask, algorithm='dilation') 

          

          sample = {'image': image,
                      'mask': mask,
                      'boundary_mask': label_boundary
                      }

          
          return sample
