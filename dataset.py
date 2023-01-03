import numpy as np
import os
import cv2
from torch.utils.data import Dataset

# TODO: Think about making this more generic and flexible, for now architecture is very
# stuck, it only works with a certain format of inputs, that's a bad smell
class CellDataset(Dataset):
  def __init__(self, images_dir, masks_dir, transform=None, classes=None):
    self.ids = os.listdir(images_dir)
    self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
    self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
    self.transform = transform
    self.classes = classes
    self.aug_images_dir = images_dir + "_AUG" 
    self.aug_masks_dir = masks_dir + "_AUG"
    self.aug_images = []
    self.aug_masks = []
    if os.path.exists(self.aug_images_dir) == False:
      os.mkdir(self.aug_images_dir)
    if os.path.exists(self.aug_masks_dir) == False:
      os.mkdir(self.aug_masks_dir)

    # convert str names to class values on masks
    self.class_values = [self.classes.index(cls) for cls in classes]
  def __len__(self):
    return len(self.ids)
    
  def __getitem__(self, i):
          
          # read data
          image = cv2.imread(self.images_fps[i])
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          mask = cv2.imread(self.masks_fps[i], 0)
          
          # extract certain classes from mask (e.g. cars)
          masks = [(mask == v) for v in self.class_values]
          mask = np.stack(masks, axis=-1).astype('float')
          
          # apply augmentations
          if self.augmentation:
              sample = self.augmentation(image=image, mask=mask)
              image, mask = sample['image'], sample['mask']
          
          # apply preprocessing
          if self.preprocessing:
              sample = self.preprocessing(image=image, mask=mask)
              image, mask = sample['image'], sample['mask']
              
          return image, mask
# Before reaching augmentations we should first threshold masks properly into different classes
# According to each cell type, it can be in a simple ascending order
# (i.e cell A has mask with 1's and 0's, cell B has mask with 2's and 0's...)
  def __apply__(self, images_to_generate):
    for img_dir, mask_dir in zip(self.image_dirs, self.mask_dirs):
      mask_class = 1
      for index in range(0, images_to_generate):
        img_path = os.path.join(img_dir, self.images[img_dir][index % self.__len__(img_dir)])
        mask_path = os.path.join(mask_dir, self.masks[mask_dir][index % self.__len__(img_dir)])
        image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # TODO: Change this part later to threshold to different values for each class of cell
        # (I.E cell A will threshold mask with 1, cell B with 2...)
        ret, mask = cv2.threshold(mask, 1, mask_class, cv2.THRESH_BINARY)

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
      mask_class = mask_class + 1
  
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


