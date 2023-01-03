#import tensorflow as tf
from dataset import CellDataset
import os
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
):
  train_ds = CellDataset(
    image_dirs=train_img_dir,
    mask_dirs=train_mask_dir,
    transform=train_transform,
    )
  return train_ds

def add_class_to_image_name(dataset_names, dir_list, dst_dir):
  for d_name in dataset_names:
    for directory in dir_list:
      if d_name in directory:
        for filename in os.listdir(directory):
          src = f"{directory}/{filename}"
          dst = f"{dst_dir}/{d_name}_{filename}"
          os.rename(src, dst)


  
