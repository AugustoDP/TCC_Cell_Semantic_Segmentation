import numpy as np
import albumentations as albu
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import os
import random




###############################################################################
# This is the dataset that Cut mix samples images and masks from
###############################################################################
dataset_img_dict = {'DIC-C2DH-HeLa' : ['/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01', '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/02'],
                 'Fluo-C2DL-MSC' : ['/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/01', '/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/02'],
                 'Fluo-N2DH-GOWT1' : ['/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/01', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/02'],
                 }
images_fp_dict = {'DIC-C2DH-HeLa' : [],
                 'Fluo-C2DL-MSC' : [],
                 'Fluo-N2DH-GOWT1' : [],
                 }
img_list = []
for key in dataset_img_dict:
  for item in dataset_img_dict[key]:
    img_list = os.listdir(item)
    img_list.sort()
    img_list = [os.path.join(item, img) for img in img_list]
    images_fp_dict[key] += (img_list)

dataset_mask_dict = {'DIC-C2DH-HeLa' : ['/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/02_ST/SEG'],
                 'Fluo-C2DL-MSC' : ['/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/02_ST/SEG'],
                 'Fluo-N2DH-GOWT1' : ['/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/02_ST/SEG'],
                 }
masks_fp_dict = {'DIC-C2DH-HeLa' : [],
                 'Fluo-C2DL-MSC' : [],
                 'Fluo-N2DH-GOWT1' : [],
                 }
mask_list = []
for key in dataset_mask_dict:
  for item in dataset_mask_dict[key]:
    mask_list = os.listdir(item)
    mask_list.sort()
    mask_list = [os.path.join(item, img) for img in mask_list]
    masks_fp_dict[key] += (mask_list)

###############################################################################
# Here we define the Random Local Rotate custom augmentation
###############################################################################

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def circle_rotate(image, **kwargs):

    box = (x-radius,y-radius,x+radius+1,y+radius+1)
    
    if image.shape == (256,256, 1):
      image = np.squeeze(image, axis=2)  
      img = Image.fromarray(image)
      background = Image.new("I;16", img.size, 0)
      mask = np.zeros(image.shape[:2], dtype="uint8")
      cv2.circle(mask, (x, y), radius, 255, -1)
      cv2_img = cv2.bitwise_and(image, image, mask=mask)
      crop = cv2_img[y-radius:y+radius+1, x-radius:x+radius+1]
      crop = rotate_image(crop, degree)   
      crop = Image.fromarray(crop)
      background.paste(crop, (x-radius, y-radius)) 
      back = np.array(background)
      comb_img = image.copy()
      comb_img[mask>0] = back[mask>0]
      comb_img = np.expand_dims(comb_img, axis=-1)
      new_img = comb_img
    else:
      img = Image.fromarray(image)
      background = Image.new("RGBA", img.size, (0,0,0,0))
      mask = Image.new("RGBA", img.size, 0)
      draw = ImageDraw.Draw(mask)
      draw.ellipse(box, fill='green', outline=None)
      new_img = Image.composite(img, background, mask)
      crop = new_img.crop(box=box)
      crop = crop.rotate(degree)
      background.paste(crop, (x-radius, y-radius))
      new_img = Image.composite(background, img, mask)
    return np.array(new_img)

###############################################################################
# Here we define helper and random seeds for cutmix and RLR augmentations
###############################################################################

def rand_bbox(size, lamb):
    """ Generate random bounding box 
    Args:
        - size: [width, breadth] of the bounding box
        - lamb: (lambda) cut ratio parameter, sampled from Beta distribution
    Returns:
        - Bounding box
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


#cutmix random seeds
height=np.random.randint(60, 200)
width=np.random.randint(60, 200)
beta = 1
lam = np.random.beta(beta, beta)
bbx1, bby1, bbx2, bby2 = rand_bbox((256,256), lam)
rand_dataset = np.random.choice(list(images_fp_dict.keys()))
rand_index =  np.random.choice(len(images_fp_dict[rand_dataset]))

#circle rotate random seeds
radius=np.random.randint(60, 121)
degree=np.random.randint(-90, 91)
x=np.random.randint(radius, 257-radius)
y=np.random.randint(radius, 257-radius)

###############################################################################
# Here we define the Cut-Mix custom augmentation
###############################################################################


def randomize_custom_aug_seeds():
  height=np.random.randint(60, 200)
  width=np.random.randint(60, 200)
  beta = 1
  lam = np.random.beta(beta, beta)
  bbx1, bby1, bbx2, bby2 = rand_bbox((256,256), lam)

  rand_dataset = np.random.choice(list(images_fp_dict.keys()))
  rand_index =  np.random.choice(len(images_fp_dict[rand_dataset]))

  #circle rotate random seeds
  radius=np.random.randint(60, 121)
  degree=np.random.randint(-90, 91)
  x=np.random.randint(radius, 257-radius)
  y=np.random.randint(radius, 257-radius)


def generate_cutmix_image(image, **kwargs):
    """ Generate a CutMix augmented image from a batch 
    Args:
      - image_batch: a batch of input images
      - image_batch_labels: labels corresponding to the image batch
      - beta: a parameter of Beta distribution.
    Returns:
      - CutMix image batch, updated labels
    """
    
    # copy image to transform
    image_updated = image
    # Change operation accordingly if input image is mask or regular RGB image
    if image.shape == (256, 256, 1):
      # get image to sample cut from
      image_to_cut = cv2.imread(masks_fp_dict[rand_dataset][rand_index], cv2.IMREAD_UNCHANGED)
      image_to_cut = cv2.resize(image_to_cut, (256, 256))
      image_to_cut = np.expand_dims(image_to_cut, axis=-1)
      # paste cut to image
      image_updated[bbx1:bbx2, bby1:bby2,:] = image_to_cut[bbx1:bbx2, bby1:bby2, :]
    else:
      # get image to sample cut from
      image_to_cut = cv2.imread(images_fp_dict[rand_dataset][rand_index])
      image_to_cut = cv2.resize(image_to_cut, (256, 256))
      # paste cut to image
      image_updated[bbx1:bbx2, bby1:bby2, :] = image_to_cut[bbx1:bbx2, bby1:bby2, :]
    
    return image_updated
