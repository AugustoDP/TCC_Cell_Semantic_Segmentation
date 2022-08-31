import numpy as np
import os
import cv2
from torch.utils.data import Dataset


class CellDataset(Dataset):
  def __init__(self, image_dirs, mask_dirs, transform=None):
    self.image_dirs = image_dirs
    self.mask_dirs = mask_dirs
    self.transform = transform
    self.images = os.listdir(image_dirs)
    self.masks = os.listdir(mask_dirs)
    self.aug_image_dirs = self.image_dirs + "_AUG"
    self.aug_mask_dirs = self.mask_dirs + "_AUG"
    self.aug_images = []
    self.aug_masks = []
    if os.path.exists(self.aug_image_dirs) == False:
      os.mkdir(self.aug_image_dirs)
    if os.path.exists(self.aug_mask_dirs) == False:
      os.mkdir(self.aug_mask_dirs)
    self.images.sort()
    self.masks.sort()

  def __len__(self):
    return len(self.images)

  def __apply__(self, images_to_generate):
    for index in range(0, images_to_generate):
      img_path = os.path.join(self.image_dirs, self.images[index % self.__len__()])
      mask_path = os.path.join(self.mask_dirs, self.masks[index % self.__len__()])
      image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
      ret, mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)

      if self.transform is not None:
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

      image_name = os.path.basename(self.images[index % self.__len__()])
      mask_name = os.path.basename(self.masks[index % self.__len__()])
      new_image_name = "%s_%s.png" %(image_name[:-4], index)
      new_mask_name = "%s_%s.png" %(mask_name[:-4], index)
      os.chdir(self.aug_image_dirs)  
      cv2.imwrite(new_image_name, image)
      os.chdir(self.aug_mask_dirs)   
      cv2.imwrite(new_mask_name, mask)
  
  def __read_augmented__(self):
    self.aug_images = os.listdir(self.aug_image_dirs)
    self.aug_masks = os.listdir(self.aug_mask_dirs)
    self.aug_images.sort()
    self.aug_masks.sort()  

  def __get_img_mask_list__(self, height=256, width=256):
    imgs_list = []
    masks_list = []
    for index in range(0, len(self.aug_images)):
      img_path = os.path.join(self.aug_image_dirs, self.aug_images[index])
      mask_path = os.path.join(self.aug_mask_dirs, self.aug_masks[index])
      image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
      image = cv2.resize(image, (height, width))
      mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
      mask = cv2.resize(mask, (height, width))
      mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      imgs_list.append(image)
      masks_list.append(mask)
    imgs_list = np.array(imgs_list)
    masks_list = np.array(masks_list)
    return imgs_list, masks_list


