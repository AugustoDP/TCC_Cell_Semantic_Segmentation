import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
import cv2
from os import path
from scipy.ndimage import rotate

import albumentations as A
images_to_generate=1000




images_path="/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01" #path to original images
masks_path = "/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST/SEG"


img_augmented_path="/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_AUG" # path to store aumented images
msk_augmented_path="/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST_AUG" # path to store aumented images
if path.exists(img_augmented_path) == False:
  os.mkdir(img_augmented_path)
if path.exists(msk_augmented_path) == False:
  os.mkdir(msk_augmented_path)
images=[] # to store paths of images from folder
masks=[]

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

for msk in os.listdir(masks_path):  # read image name from folder and append its path into "images" array     
    masks.append(os.path.join(masks_path,msk))


aug = A.Compose([
    A.VerticalFlip(p=0.5),              
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=1),
    A.Transpose(p=1),
    #A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
    A.GridDistortion(p=1)
    ]
)

for i in range(0, images_to_generate):
    image = images[i % len(images)]
    mask = masks[i % len(images)]
    print(image, mask)
    original_image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
    original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2RGB)
    
    augmented = aug(image=original_image, mask=original_mask)
    transformed_image = augmented['image']
    transformed_mask = augmented['mask']

     
 
    new_image_name = "augmented_image_%s.png" %(i)
    new_mask_name = "augmented_mask_%s.png" %(i)
    os.chdir(img_augmented_path)  
    cv2.imwrite(new_image_name, transformed_image)
    os.chdir(msk_augmented_path)   
    cv2.imwrite(new_mask_name, transformed_mask)
