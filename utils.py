import tensorflow as tf
from dataset import CellDataset
from tensorflow import keras



def save_checkpoint(state, filename='modelUNET.h5'):
  print("=> Saving checkpoint")
  state.save(filename)

def load_checkpoint(checkpoint, model):
  print("=> Loading checkpoint")
  model = keras.model.load_model(checkpoint)


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




  
