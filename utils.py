#import tensorflow as tf
from dataset import CellDataset
import shutil
import numpy as np
import os
import cv2
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
  batch_size,
  train_transform,
  train_classes,
  train_preprocessing
):
  train_ds = CellDataset(
    images_dir=train_img_dir,
    masks_dir=train_mask_dir,
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
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        ret, mask = cv2.threshold(mask, 1, dataset_names.index(d_name)+1, cv2.THRESH_BINARY)
        cv2.imwrite(mask_fp, mask)


def split_train_val_set(img_dir, mask_dir, split):
  img_list = os.listdir(img_dir)
  mask_list = os.listdir(mask_dir)
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