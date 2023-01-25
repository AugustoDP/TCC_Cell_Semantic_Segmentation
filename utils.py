#import tensorflow as tf
from dataset import CellDataset
import shutil
import numpy as np
import os
import cv2
from scipy import ndimage
#from tensorflow import keras
from random import shuffle
from math import floor

def save_checkpoint(state, filepath):
  print("=> Saving checkpoint")
  #state.save(filepath)

def load_checkpoint(checkpoint, model, load_compile):
  print("=> Loading checkpoint")
  #model = keras.model.load_model(checkpoint, compile=load_compile)


def get_loaders(
  train_img_dir,
  train_mask_dir,
  train_boundary_dir,
  batch_size,
  max_size,
  train_transform,
  train_classes,
  train_preprocessing
):
  train_ds = CellDataset(
    images_dir=train_img_dir,
    masks_dir=train_mask_dir,
    boundary_dir=train_boundary_dir,
    size=max_size,
    transform=train_transform,
    classes=train_classes,
    preprocessing=train_preprocessing
    )
  return train_ds

def add_class_to_image_name(dataset_names, dir_list, dst_dir):
  for d_name in dataset_names:
    for directory in dir_list:
      if d_name in directory:
        for filename in os.listdir(directory):          
          src = f"{directory}/{filename}"
          dst = f"{dst_dir}/{d_name}_{filename}"
          shutil.copy(src, dst)

def threshold_masks(dataset_names, mask_dir):
  os.chdir(mask_dir)
  for d_name in dataset_names:    
    for mask_fp in os.listdir(mask_dir):
      if d_name in mask_fp:
        mask = cv2.imread(mask_fp, cv2.IMREAD_UNCHANGED)
        ret, mask = cv2.threshold(mask, 1, dataset_names.index(d_name)+1, cv2.THRESH_BINARY)
        cv2.imwrite(mask_fp, mask)


def split_train_val_set(img_dir, mask_dir, boundary_dir, split):
  img_list = os.listdir(img_dir)
  mask_list = os.listdir(mask_dir)
  boundary_list = os.listdir(boundary_dir)
  img_list.sort()
  mask_list.sort()
  boundary_list.sort()
  # Shuffle lists together to random order
  img_list, mask_list, boundary_list = randomize_paired_file_lists(img_list, mask_list, boundary_list)
  # Split lists into train and validation sets
  train_img, val_img, train_mask, val_mask, train_boundary, val_boundary = get_training_and_validation_sets(split, img_list, mask_list, boundary_list)
  # Make directories for train and validation for img, mask and boundary
  train_img_dir, val_img_dir = make_train_val_dirs(img_dir, train_img, val_img)
  train_mask_dir, val_mask_dir = make_train_val_dirs(mask_dir, train_mask, val_mask)
  train_boundary_dir, val_boundary_dir = make_train_val_dirs(boundary_dir, train_boundary, val_boundary)
  return train_img_dir, val_img_dir, train_mask_dir, val_mask_dir, train_boundary_dir, val_boundary_dir

def randomize_paired_file_lists(img_list, mask_list, boundary_list):
    file_list_pair = list(zip(img_list, mask_list, boundary_list))
    shuffle(file_list_pair)

    a, b, c = zip(*file_list_pair)
    return a, b, c

def get_training_and_validation_sets(split=0.7, img_list=[], mask_list=[], boundary_list=[]):
    split_index = floor(len(img_list) * split)
    train_img = img_list[:split_index]
    val_img = img_list[split_index:]
    train_mask = mask_list[:split_index]
    val_mask = mask_list[split_index:]
    train_boundary = boundary_list[:split_index]
    val_boundary = boundary_list[split_index:]
    return train_img, val_img, train_mask, val_mask, train_boundary, val_boundary


def make_train_val_dirs(main_dir, train_files, validation_files):
  os.chdir(main_dir)
  os.mkdir("TRAIN")
  for train_file in train_files:
    dst = f"TRAIN/{train_file}"
    os.rename(train_file, dst)
  os.mkdir("VAL")
  for val_file in validation_files:
    dst = f"VAL/{val_file}"
    os.rename(val_file, dst)

  train_dir = os.path.join(main_dir, "TRAIN")
  val_dir = os.path.join(main_dir, "VAL")
  return train_dir, val_dir


def generate_augmented_images(
  image_dir, 
  mask_dir, 
  boundary_dir,
  augmentation_ratio,
  transform
  ):
  """ Generate a number of augmented images using specified transform.
  :param image_dir: images directory.
    :type str:
  :param mask_dir: masks directory.
    :type str:
  :param augmentation_ratio: number of augmentations to generate per image.
    :type int:
  :param transform: albumentation transform containing augmentations to perform
    :type:
  :return: None.
  """
  img_list = os.listdir(image_dir)
  mask_list = os.listdir(mask_dir)
  boundary_list = os.listdir(boundary_dir)
  img_list.sort()
  mask_list.sort()
  boundary_list.sort()
  for index in range(len(img_list)):
    for i in range(augmentation_ratio):
      # Read images
      image = cv2.imread(f"{image_dir}/{img_list[index]}")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask = cv2.imread(f"{mask_dir}/{mask_list[index]}", cv2.IMREAD_UNCHANGED)
      boundary = cv2.imread(f"{boundary_dir}/{boundary_list[index]}", cv2.IMREAD_UNCHANGED)
      # Apply transformations
      if transform is not None:
        augmented = transform(image=image, mask=mask, boundary=boundary)
        image = augmented["image"]
        mask = augmented["mask"]
        boundary = augmented["boundary"]

      # Rename augmentations
      image_name = os.path.basename(img_list[index])
      mask_name = os.path.basename(mask_list[index])
      boundary_name = os.path.basename(boundary_list[index])
      new_image_name = "%s_%s.png" %(image_name[:-4], i)
      new_mask_name = "%s_%s.png" %(mask_name[:-4], i)
      new_boundary_name = "%s_%s.png" %(boundary_name[:-4], i)
      # Save augmentations
      os.chdir(image_dir)  
      cv2.imwrite(new_image_name, image)
      os.chdir(mask_dir)   
      cv2.imwrite(new_mask_name, mask)
      os.chdir(boundary_dir)   
      cv2.imwrite(new_boundary_name, boundary)


def get_cell_ids(mask):
    """ Get cell ids in mask image.

    :param mask: mask image containing cell instances.
        :type image:
    :return: List of cell ids.
    """

    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]

    return cell_ids  

def binary_label(label):
    """ Binary label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :return: Binary label image.
    """
    return label > 0


def boundary_label_2d(label, algorithm='dilation'):
    """ Boundary label image creation.

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param algorithm: canny or dilation-based boundary creation.
        :type algorithm: str
    :return: Boundary label image.
    """

    label_bin = binary_label(label)

    if algorithm == 'canny':

        if len(get_cell_ids(label)) > 255:
            raise Exception('Canny method works only with uint8 images but more than 255 nuclei detected.')

        boundary = cv2.Canny(label.astype(np.uint8), 1, 1) > 0
        label_boundary = np.maximum(label_bin, 2 * boundary)

    elif algorithm == 'dilation':

        kernel = np.ones(shape=(3, 3), dtype=np.uint8)

        # Pre-allocation
        boundary = np.zeros(shape=label.shape, dtype=np.bool)

        nucleus_ids = get_cell_ids(label)

        for nucleus_id in nucleus_ids:
            nucleus = (label == nucleus_id)
            nucleus_boundary = ndimage.binary_dilation(nucleus, kernel) ^ nucleus
            boundary += nucleus_boundary

        label_boundary = np.maximum(label_bin, 2 * boundary)

    return label_boundary

def create_boundary_masks(mask_dir, dst_dir):
  mask_list = os.listdir(mask_dir)
  mask_list.sort()
  os.chdir(dst_dir) 
  for mask_filename in mask_list:
    mask_fp = f"{mask_dir}/{mask_filename}"
    mask = cv2.imread(mask_fp, cv2.IMREAD_UNCHANGED)
    label_boundary = boundary_label_2d(label=mask, algorithm='dilation') 
    boundary_filename = f"{dst_dir}/b_{mask_filename}" 
    cv2.imwrite(boundary_filename, label_boundary)