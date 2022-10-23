import numpy as np
import os
import cv2
from torch.utils.data import Dataset

# TODO: Think about making this more generic and flexible, for now architecture is very
# stuck, it only works with a certain format of inputs, that's a bad smell
class CellDataset(Dataset):
  def __init__(self, image_dirs, mask_dirs, transform=None):
    self.image_dirs = image_dirs
    self.mask_dirs = mask_dirs
    self.transform = transform
    self.images = {img_dir: os.listdir(img_dir) for img_dir in self.image_dirs} 
    self.masks = {mask_dir: os.listdir(mask_dir) for mask_dir in self.mask_dirs}
    self.aug_image_dirs = [img_dir + "_AUG" for img_dir in self.image_dirs]
    self.aug_mask_dirs = [mask_dir + "_AUG" for mask_dir in self.mask_dirs]
    self.aug_images = {}
    self.aug_masks = {}
    for aug_img_dir in self.aug_image_dirs:
      if os.path.exists(aug_img_dir) == False:
        os.mkdir(aug_img_dir)
    for aug_mask_dir in self.aug_mask_dirs:
      if os.path.exists(aug_mask_dir) == False:
        os.mkdir(aug_mask_dir)
    for img_dir in self.images:
      self.images[img_dir].sort()
    for mask_dir in self.masks:
      self.masks[mask_dir].sort()

  def __len__(self, img_dir):
    return len(self.images[img_dir])

# Before reaching augmentations we should first threshold masks properly into different classes
# According to each cell type, it can be in a simple ascending order
# (i.e cell A has mask with 1's and 0's, cell B has mask with 2's and 0's...)
  def __apply__(self, images_to_generate):
    for img_dir, mask_dir in zip(self.image_dirs, self.mask_dirs):
      for index in range(0, images_to_generate):
        img_path = os.path.join(img_dir, self.images[img_dir][index % self.__len__(img_dir)])
        mask_path = os.path.join(mask_dir, self.masks[mask_dir][index % self.__len__(img_dir)])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # TODO: Change this part later to threshold to different values for each class of cell
        # (I.E cell A will threshold mask with 1, cell B with 2...)
        ret, mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)

        if self.transform is not None:
          augmented = self.transform(image=image, mask=mask)
          image = augmented["image"]
          mask = augmented["mask"]

        image_name = os.path.basename(self.images[img_dir][index % self.__len__(img_dir)])
        mask_name = os.path.basename(self.masks[mask_dir][index % self.__len__(img_dir)])
        new_image_name = "%s_%s.png" %(image_name[:-4], index)
        new_mask_name = "%s_%s.png" %(mask_name[:-4], index)
        os.chdir(img_dir + "_AUG")  
        cv2.imwrite(new_image_name, image)
        os.chdir(mask_dir + "_AUG")   
        cv2.imwrite(new_mask_name, mask)
  
  def __read_augmented__(self):
    self.aug_images = {aug_img_dir: os.listdir(aug_img_dir) for aug_img_dir in self.aug_image_dirs}
    self.aug_masks = {aug_mask_dir: os.listdir(aug_mask_dir) for aug_mask_dir in self.aug_mask_dirs}
    for aug_img_dir in self.aug_images:
      self.aug_images[aug_img_dir].sort()
    for aug_mask_dir in self.aug_masks:
      self.aug_masks[aug_mask_dir].sort()  

  def __get_img_mask_list__(self, height=256, width=256):
    imgs_list = []
    masks_list = []
    for aug_img_dir, aug_mask_dir in zip(self.aug_image_dirs, self.aug_mask_dirs):
      for index in range(0, len(self.aug_images[aug_img_dir])):
        img_path = os.path.join(aug_img_dir, self.aug_images[aug_img_dir][index])
        mask_path = os.path.join(aug_mask_dir, self.aug_masks[aug_mask_dir][index])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (height, width))
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (height, width))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        imgs_list.append(image)
        masks_list.append(mask)
    imgs_list = np.array(imgs_list)
    masks_list = np.array(masks_list)
    print(len(imgs_list), len(masks_list))
    return imgs_list, masks_list


