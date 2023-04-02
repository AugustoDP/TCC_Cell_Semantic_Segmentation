  #import tensorflow as tf
import logging
import segmentation_models_pytorch as sm
import glob
import cv2
import os
from os import path
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
import shutil

from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from kornia.losses import focal_loss
from matplotlib import pyplot as plt

from metrics import dice_loss
from eval import eval_net
from augmentation import (
  get_training_augmentation,
  get_validation_augmentation,
  get_preprocessing,
)
from utils import (
  save_checkpoint,
  load_checkpoint,
  get_loaders,
  add_class_to_image_name,
  threshold_masks,
  split_train_val_set,
  generate_augmented_images
)

MAIN_IMAGE_DIR = '/content/TCC_Cell_Semantic_Segmentation/IMAGES'
MAIN_MASK_DIR = '/content/TCC_Cell_Semantic_Segmentation/MASKS'
MAIN_TEST_IMAGE_DIR = '/content/TCC_Cell_Semantic_Segmentation/TEST_IMAGES'
MAIN_TEST_MASK_DIR = '/content/TCC_Cell_Semantic_Segmentation/TEST_MASKS'
DATASET_NAMES = ['DIC-C2DH-HeLa']
TESTSET_NAMES = ['Fluo-N2DL-HeLa']
TRAIN_IMG_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01', '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/02']
TRAIN_MASK_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/02_ST/SEG']
TEST_IMG_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DL-HeLa/01', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DL-HeLa/02']
TEST_MASK_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DL-HeLa/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DL-HeLa/02_ST/SEG']
BATCH_SIZE = 8
EPOCHS = 150
LR = 0.001
LOAD_MODEL = False
TRAIN_MODEL = True
IMAGE_SIZE = 256
OPTIMIZER = 'Adam'
LOSS = sm.utils.losses.DiceLoss()
METRICS = [sm.utils.metrics.IoU(threshold=0.5),]
BACKBONE = 'timm-efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
#AUGMENTATION_PER_IMAGE = 8
TRAIN_VAL_SPLIT = 0.8
TEST_IMG = ''
NUM_CLASSES = 1
ACTIVATION = "sigmoid"
TEST_MODEL = False
MODEL_PATH = '/content/TCC_Cell_Semantic_Segmentation/ResultsCP_final_epoch.pth'
RESULTS_PATH = '/content/TCC_Cell_Semantic_Segmentation/Results'

def train_model(model, 
            device,
            training_set,
            validation_set,
            epochs=50,
            n_classes=1,  
):
  train_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
  valid_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
  # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
  loss = sm.utils.losses.DiceLoss()
  metrics = [
      sm.utils.metrics.IoU(threshold=0.5),
  ]

  optimizer = torch.optim.Adam([ 
      dict(params=model.parameters(), lr=0.001),
  ])
  # create epoch runners 
  # it is a simple loop of iterating over dataloader`s samples
  train_epoch = sm.utils.train.TrainEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      optimizer=optimizer,
      device=device,
      verbose=True,
  )

  valid_epoch = sm.utils.train.ValidEpoch(
      model, 
      loss=loss, 
      metrics=metrics, 
      device=device,
      verbose=True,
  )
  # train model for x epochs

  max_score = 0
  current_loss = 0
  last_loss = -1
  stop_count = 0
  for i in range(0, epochs):
      #Stop training early to avoid overfit
      if stop_count == 5:
        print('Loss is not decreasing! Stopping training')
        break

      print('\nEpoch: {}'.format(i))
      train_logs = train_epoch.run(train_loader)
      valid_logs = valid_epoch.run(valid_loader)
      
      # Check if loss is changing
      if last_loss == -1:
        last_loss = valid_logs['dice_loss']
      else:
        current_loss = valid_logs['dice_loss']
        if last_loss - current_loss < 0.001:
          stop_count += 1
        else:
          stop_count = 0
        last_loss = current_loss

      # do something (save model, change lr, etc.)
      if max_score < valid_logs['iou_score']:
          max_score = valid_logs['iou_score']          
          try:
            os.mkdir(RESULTS_PATH)
            logging.info('Created checkpoint directory')
          except OSError:
              pass
          torch.save(model.state_dict(),
                      RESULTS_PATH + f'CP_epoch{i + 1}.pth')
          logging.info(f'Checkpoint {i + 1} saved !')
      # if i == 15:
      #     optimizer.param_groups[0]['lr'] = 5e-5
      #     print('Decrease decoder learning rate to 5e-5!')
      # if i == 50:
      #     optimizer.param_groups[0]['lr'] = 1e-5
      #     print('Decrease decoder learning rate to 1e-5!')
  torch.save(model.state_dict(),
                      RESULTS_PATH + f'CP_final_epoch.pth')
  logging.info(f'Last Checkpoint saved !')
# def predict(model, image_path):
#   image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)       
#   image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
#   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#   image = np.expand_dims(image, axis=0)
#   prediction = model.predict(image)
#   #View and Save segmented image  
#   prediction_image = np.argmax(prediction, axis=3)[0,:,:]
#   #prediction_image = prediction.reshape([256,256])
#   #new_img = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)
#   plt.imshow(prediction_image, cmap='gray')
#   prediction_image_name = 'test_' + os.path.basename(image_path[:-4]) + '.png' 
#   output_path = os.path.join(RESULTS_PATH, prediction_image_name)
#   plt.imsave(output_path, prediction_image, cmap='gray')

def test_model(best_model, 
            device,
            test_set,
            ):

  test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
  # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
  loss = sm.utils.losses.DiceLoss()
  metrics = [
      sm.utils.metrics.IoU(threshold=0.5),
  ]
    # evaluate model on test set
  test_epoch = sm.utils.train.ValidEpoch(
      model=best_model,
      loss=loss,
      metrics=metrics,
      device=device,
      verbose=True,
  )

  logs = test_epoch.run(test_loader)          


def main():

  logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
  # Determine device
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info(f'Using device {device}')

  preprocessing_fn = sm.encoders.get_preprocessing_fn(BACKBONE, ENCODER_WEIGHTS)

  if os.path.exists(MAIN_IMAGE_DIR):
    shutil.rmtree(MAIN_IMAGE_DIR)
  os.mkdir(MAIN_IMAGE_DIR)
  if os.path.exists(MAIN_MASK_DIR):
    shutil.rmtree(MAIN_MASK_DIR)
  os.mkdir(MAIN_MASK_DIR)

  if os.path.exists(MAIN_TEST_IMAGE_DIR):
    shutil.rmtree(MAIN_TEST_IMAGE_DIR)
  os.mkdir(MAIN_TEST_IMAGE_DIR)
  if os.path.exists(MAIN_TEST_MASK_DIR):
    shutil.rmtree(MAIN_TEST_MASK_DIR)
  os.mkdir(MAIN_TEST_MASK_DIR)

  model = sm.EfficientUnetPlusPlus(BACKBONE, encoder_weights=None, classes=NUM_CLASSES, activation=ACTIVATION)


  # Distribute training over GPUs
  model = nn.DataParallel(model)

  # If want to load model or train
  if LOAD_MODEL:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
  if TRAIN_MODEL:
    add_class_to_image_name(DATASET_NAMES,
     TRAIN_IMG_DIRS, 
     MAIN_IMAGE_DIR, 
     TRAIN_MASK_DIRS, 
     MAIN_MASK_DIR)
    threshold_masks(DATASET_NAMES, MAIN_MASK_DIR)
    train_img_dir, val_img_dir, train_mask_dir, val_mask_dir = split_train_val_set(MAIN_IMAGE_DIR, MAIN_MASK_DIR, TRAIN_VAL_SPLIT)
    #generate_augmented_images(train_img_dir, train_mask_dir, AUGMENTATION_PER_IMAGE, get_training_augmentation())
    train_ds = get_loaders(
        train_img_dir=train_img_dir,
        train_mask_dir=train_mask_dir,
        batch_size=BATCH_SIZE,
        max_size=IMAGE_SIZE,
        train_transform=get_training_augmentation(),
        train_classes=DATASET_NAMES,
        train_preprocessing=get_preprocessing(preprocessing_fn)
        )
    val_ds = get_loaders(
      train_img_dir=val_img_dir,
      train_mask_dir=val_mask_dir,
      batch_size=BATCH_SIZE,
      max_size=IMAGE_SIZE,
      train_transform=None,
      train_classes=DATASET_NAMES,
      train_preprocessing=get_preprocessing(preprocessing_fn)
      )

    train_model(model=model, 
              device=device,
              training_set=train_ds,
              validation_set=val_ds,
              epochs=EPOCHS,
              n_classes=NUM_CLASSES
              )

  # If have a model ready to test
  if TEST_MODEL:
    add_class_to_image_name(TESTSET_NAMES, 
    TEST_IMG_DIRS, 
    MAIN_TEST_IMAGE_DIR, 
    TEST_MASK_DIRS,
    MAIN_TEST_MASK_DIR)
    threshold_masks(TESTSET_NAMES, MAIN_TEST_MASK_DIR)
    test_ds = get_loaders(
        train_img_dir=MAIN_TEST_IMAGE_DIR,
        train_mask_dir=MAIN_TEST_MASK_DIR,
        batch_size=BATCH_SIZE,
        max_size=IMAGE_SIZE,
        train_transform=None,
        train_classes=TESTSET_NAMES,
        train_preprocessing=get_preprocessing(preprocessing_fn)
        )
    test_model(best_model=model, device=device, test_set=test_ds)

  


if __name__ == "__main__":
  main()

