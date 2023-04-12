import numpy as np
import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from augmentation import rand_augment
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
        # Implement additional initialization logic if needed

    def __len__(self):
        # Replace `...` with the actual implementation
        return len(self.image_ids)

    def __getitem__(self, index):
        # Implement logic to get an image and its mask using the received index.
        #
        # `image` should be a NumPy array with the shape [height, width, num_channels].
        # If an image contains three color channels, it should use an RGB color scheme.
        #
        # `mask` should be a NumPy array with the shape [height, width, num_classes] where `num_classes`
        # is a value set in the `search.yaml` file. Each mask channel should encode values for a single class (usually
        # pixel in that channel has a value of 1.0 if the corresponding pixel from the image belongs to this class and
        # 0.0 otherwise). During augmentation search, `nn.BCEWithLogitsLoss` is used as a segmentation loss.
        # read data
        image = cv2.imread(self.images_fps[index], cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (self.max_size, self.max_size))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = np.array(image).astype(np.uint16)
        image = np.uint8((image / image.max()) * 255)
        mask = cv2.imread(self.masks_fps[index], cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (self.max_size, self.max_size))
        # # extract certain classes from mask (e.g. cars)
        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        mask = np.expand_dims(mask, axis=-1).astype('float')
        # apply augmentations
        if self.transform:
            aug = rand_augment(1)
            sample = aug(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']     
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']  


        return image, mask
