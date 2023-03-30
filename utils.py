#import tensorflow as tf
from configs.dataset import CellDataset
import shutil
import numpy as np
import os
import cv2
#from tensorflow import keras
import random
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
  batch_size,
  max_size,
  train_transform,
  train_classes,
  train_preprocessing
):
  train_ds = CellDataset(
    images_dir=train_img_dir,
    masks_dir=train_mask_dir,
    size=max_size,
    transform=train_transform,
    classes=train_classes,
    preprocessing=train_preprocessing
    )
  return train_ds

def add_class_to_image_name(
 dataset_names,
 img_dir_list,
 img_dst_dir, 
 msk_dir_list, 
 msk_dst_dir
 ):
  for d_name in dataset_names:
    i = 0
    for img_directory, msk_directory in zip(img_dir_list, msk_dir_list):
      if d_name in img_directory and d_name in msk_directory:
        img_filename_list = os.listdir(img_directory)
        msk_filename_list = os.listdir(msk_directory)
        img_filename_list.sort()
        msk_filename_list.sort()
        index_list = random.sample(range(len(img_filename_list)), 48)
        for selected_index in index_list:          
          img_src = f"{img_directory}/{img_filename_list[selected_index]}"
          img_dst = f"{img_dst_dir}/{d_name}_{i}_{img_filename_list[selected_index]}"
          shutil.copy(img_src, img_dst)
          msk_src = f"{msk_directory}/{msk_filename_list[selected_index]}"
          msk_dst = f"{msk_dst_dir}/{d_name}_{i}_{msk_filename_list[selected_index]}"
          shutil.copy(msk_src, msk_dst)
      i = i + 1

def threshold_masks(dataset_names, mask_dir):
  os.chdir(mask_dir)
  for d_name in dataset_names:    
    for mask_fp in os.listdir(mask_dir):
      if d_name in mask_fp:
        mask = cv2.imread(mask_fp, cv2.IMREAD_UNCHANGED)
        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        cv2.imwrite(mask_fp, mask)


def split_train_val_set(img_dir, mask_dir, split):
  img_list = os.listdir(img_dir)
  mask_list = os.listdir(mask_dir)
  img_list.sort()
  mask_list.sort()
  img_list, mask_list = randomize_pair_file_lists(img_list, mask_list)
  training_img, validation_img, training_mask, validation_mask = get_training_and_validation_sets(split, img_list, mask_list)
  train_img_dir, val_img_dir = make_train_val_dirs(img_dir, training_img, validation_img)
  train_mask_dir, val_mask_dir = make_train_val_dirs(mask_dir, training_mask, validation_mask)
  return train_img_dir, val_img_dir, train_mask_dir, val_mask_dir



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

def randomize_pair_file_lists(img_list, mask_list):
    file_list_pair = list(zip(img_list, mask_list))
    shuffle(file_list_pair)

    a, b = zip(*file_list_pair)
    return a, b

def get_training_and_validation_sets(split=0.7, img_list=[], mask_list=[]):
    split_index = floor(len(img_list) * split)
    training_img = img_list[:split_index]
    validation_img = img_list[split_index:]
    training_mask = mask_list[:split_index]
    validation_mask = mask_list[split_index:]
    return training_img, validation_img, training_mask, validation_mask


def generate_augmented_images(
  image_dir, 
  mask_dir, 
  augmentation_ratio,
  transform
  ):
  img_list = os.listdir(image_dir)
  mask_list = os.listdir(mask_dir)
  img_list.sort()
  mask_list.sort()
  for index in range(len(img_list)):
    for i in range(augmentation_ratio):

      image = cv2.imread(f"{image_dir}/{img_list[index]}")
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mask = cv2.imread(f"{mask_dir}/{mask_list[index]}", cv2.IMREAD_UNCHANGED)
      if transform is not None:
        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

      image_name = os.path.basename(img_list[index])
      mask_name = os.path.basename(mask_list[index])
      new_image_name = "%s_%s.png" %(image_name[:-4], i)
      new_mask_name = "%s_%s.png" %(mask_name[:-4], i)
      os.chdir(image_dir)  
      cv2.imwrite(new_image_name, image)
      os.chdir(mask_dir)   
      cv2.imwrite(new_mask_name, mask)

  





