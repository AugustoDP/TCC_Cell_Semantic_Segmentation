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
  get_preprocessing
)
from utils import (
  save_checkpoint,
  load_checkpoint,
  get_loaders,
  add_class_to_image_name,
  threshold_masks,
  split_train_val_set
)

MAIN_IMAGE_DIR = '/content/TCC_Cell_Semantic_Segmentation/IMAGES'
MAIN_MASK_DIR = '/content/TCC_Cell_Semantic_Segmentation/MASKS'
DATASET_NAMES = ['Fluo-C2DL-MSC', 'Fluo-N2DH-GOWT1']
TRAIN_IMG_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/01', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/01']
TRAIN_MASK_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/01_ST/SEG']
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
LOAD_MODEL = False
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OPTIMIZER = 'Adam'
LOSS = sm.losses.DiceLoss
METRICS = "accuracy"#sm.metrics.iou_score
BACKBONE = 'timm-efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
AUGMENTATION_PER_IMAGE = 8
TRAIN_SPLIT_SIZE = 0.8
TEST_IMG = '/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/02/t000.tif'
RESULTS_PATH ="/content/TCC_Cell_Semantic_Segmentation/Results" # path to store model results
NUM_CLASSES = 2
ACTIVATION = "softmax"

def   train_fn(model, 
            device,
            training_set,
            validation_set,
            dir_checkpoint,
            epochs=50,
            batch_size=2,
            lr=0.001,
            save_cp=True,
            img_scale=1,
            n_classes=2,
            n_channels=3,
            augmentation_ratio = 8):

  train = training_set 
  val = validation_set
  n_train = len(train)
  n_val = len(val)
  train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
  val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


  # Sets the effective batch size according to the batch size and the data augmentation ratio
  batch_size = (1 + augmentation_ratio)*batch_size

  # Prepares the summary file
  writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
  global_step = 0

  logging.info(f'''Starting training:
      Epochs:          {epochs}
      Batch size:      {batch_size}
      Learning rate:   {lr}
      Training size:   {n_train}
      Validation size: {n_val}
      Checkpoints:     {save_cp}
      Device:          {device.type}
      Images scaling:  {img_scale}
      Augmentation ratio: {augmentation_ratio}
  ''')

  # Choose the optimizer and scheduler 
  optimizer = optim.Adam(model.parameters(), lr=lr)
  scheduler = optim.lr_scheduler.StepLR(optimizer, epochs//3, gamma=0.1, verbose=True)

  # Train loop
  for epoch in range(epochs):
      model.train()
      epoch_loss = 0
      with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
          for batch in train_loader:
              imgs = batch['image']
              true_masks = batch['mask']
              # DataLoaders return lists of tensors. TODO: Concatenate the lists inside the DataLoaders
              imgs = torch.cat(imgs, dim = 0)
              true_masks = torch.cat(true_masks, dim = 0)

              assert imgs.shape[1] == n_channels, \
                  f'Network has been defined with {n_channels} input channels, ' \
                  f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                  'the images are loaded correctly.'

              imgs = imgs.to(device=device, dtype=torch.float32)
              mask_type = torch.float32 if n_classes == 1 else torch.long
              true_masks = true_masks.to(device=device, dtype=mask_type)

              masks_pred = model(imgs)                         
              
              # Compute loss
              loss = focal_loss(masks_pred, true_masks.squeeze(1), alpha=0.25, gamma = 2, reduction='mean').unsqueeze(0)
              loss += dice_loss(masks_pred, true_masks.squeeze(1), True, k = 0.75)

              epoch_loss += loss.item()
              writer.add_scalar('Loss/train', loss.item(), global_step)
              pbar.set_postfix(**{'loss (batch)': loss.item()})

              optimizer.zero_grad()
              loss.backward()
              nn.utils.clip_grad_value_(model.parameters(), 0.1)
              optimizer.step()

              pbar.update(imgs.shape[0]//(1 + augmentation_ratio))
              global_step += 1
              if global_step % (n_train // (batch_size / (1 + augmentation_ratio))) == 0:
                  for tag, value in model.named_parameters():
                      tag = tag.replace('.', '/')
                      
                      try:
                          writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                          writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                      except:
                          pass
                  
                  epoch_score = eval_net(model, train_loader, device)
                  val_score = eval_net(model, val_loader, device)
                  writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                  if n_classes > 1:
                      logging.info('Validation loss: {}'.format(val_score))
                      writer.add_scalar('Generalized dice loss/train', epoch_score, global_step)
                      writer.add_scalar('Generalized dice loss/test', val_score, global_step)
                  else:
                      logging.info('Validation loss: {}'.format(val_score))
                      writer.add_scalar('Dice loss/train', epoch_score, global_step)
                      writer.add_scalar('Dice loss/test', val_score, global_step)       
      scheduler.step()         
      if save_cp:
          try:
              os.mkdir(dir_checkpoint)
              logging.info('Created checkpoint directory')
          except OSError:
              pass
          torch.save(model.state_dict(),
                      dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
          logging.info(f'Checkpoint {epoch + 1} saved !')
  writer.close()


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

  add_class_to_image_name(DATASET_NAMES, TRAIN_IMG_DIRS, MAIN_IMAGE_DIR)
  add_class_to_image_name(DATASET_NAMES, TRAIN_MASK_DIRS, MAIN_MASK_DIR)
  threshold_masks(DATASET_NAMES, MAIN_MASK_DIR)
  train_img_dir, val_img_dir, train_mask_dir, val_mask_dir = split_train_val_set(MAIN_IMAGE_DIR, MAIN_MASK_DIR, TRAIN_SPLIT_SIZE)
  train_ds = get_loaders(
      train_img_dir=train_img_dir,
      train_mask_dir=train_mask_dir,
      batch_size=BATCH_SIZE,
      train_transform=get_training_augmentation(),
      train_classes=DATASET_NAMES,
      train_preprocessing=get_preprocessing(preprocessing_fn)
      )
  val_ds = get_loaders(
    train_img_dir=val_img_dir,
    train_mask_dir=val_mask_dir,
    batch_size=BATCH_SIZE,
    train_transform=get_validation_augmentation(),
    train_classes=DATASET_NAMES,
    train_preprocessing=get_preprocessing(preprocessing_fn)
    )
  # train_ds.__apply__(IMAGES_TO_GENERATE)
  # train_ds.__read_augmented__()
  model = sm.EfficientUnetPlusPlus(BACKBONE, encoder_weights=ENCODER_WEIGHTS, classes=NUM_CLASSES, activation=ACTIVATION)
  # Distribute training over GPUs
  model = nn.DataParallel(model)
  train_fn(model=model, 
            device=device,
            training_set=train_ds,
            validation_set=val_ds,
            dir_checkpoint=RESULTS_PATH,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            save_cp=True,
            img_scale=1,
            n_classes=NUM_CLASSES,
            n_channels=3,
            augmentation_ratio = AUGMENTATION_PER_IMAGE)
  # model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])
  # model = train_fn(train_ds, model)
  # save_model_path = os.path.join(RESULTS_PATH, 'modelUNET.h5')
  # save_checkpoint(model, save_model_path)
  # predict(model, TEST_IMG)


if __name__ == "__main__":
  main()

