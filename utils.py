#import tensorflow as tf
from dataset import CellDataset
import shutil
import numpy as np
import os
import cv2
#from tensorflow import keras



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
    image_dirs=train_img_dir,
    mask_dirs=train_mask_dir,
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

  
