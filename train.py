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
  generate_augmented_images,
  create_boundary_masks
)

MAIN_IMAGE_DIR = '/content/TCC_Cell_Semantic_Segmentation/IMAGES'
MAIN_MASK_DIR = '/content/TCC_Cell_Semantic_Segmentation/MASKS'
MAIN_BOUNDARY_DIR = '/content/TCC_Cell_Semantic_Segmentation/BOUNDARY'
DATASET_NAMES = ['Cell']
TRAIN_IMG_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/01', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/01', '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01']
TRAIN_MASK_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/Fluo-N2DH-GOWT1/01_ST/SEG', '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST/SEG']
BATCH_SIZE = 1
EPOCHS = 35
LR = 0.001
LOAD_MODEL = False
IMAGE_SIZE = 256
OPTIMIZER = 'Adam'
LOSS = sm.losses.DiceLoss
METRICS = "accuracy"#sm.metrics.iou_score
BACKBONE = 'timm-efficientnet-b0'
ENCODER_WEIGHTS = 'imagenet'
AUGMENTATION_PER_IMAGE = 5
TRAIN_VAL_SPLIT = 0.8
TEST_IMG = '/content/TCC_Cell_Semantic_Segmentation/Fluo-C2DL-MSC/02/t000.tif'
RESULTS_PATH ="/content/TCC_Cell_Semantic_Segmentation/Results" # path to store model results
NUM_CLASSES = 3
ACTIVATION = "softmax2d"


def train(net, datasets, configs, device, path_models):
    """ Train the model.

    :param net: Model/Network to train.
        :type net:
    :param datasets: Dictionary containing the training and the validation data set.
        :type datasets: dict
    :param configs: Dictionary with configurations of the training process.
        :type configs: dict
    :param device: Use (multiple) GPUs or CPU.
        :type device: torch device
    :param path_models: Path to the directory to save the models.
        :type path_models: pathlib Path object
    :return: None
    """

    print('-' * 20)
    print('Train {0} on {1} images, validate on {2} images'.format(configs['run_name'],
                                                                   len(datasets['train']),
                                                                   len(datasets['val'])))

    # Data loader for training and validation set
    apply_shuffling = {'train': True, 'val': False}
    dataloader = {x: torch.utils.data.DataLoader(datasets[x],
                                                 batch_size=configs['batch_size'],
                                                 shuffle=apply_shuffling,
                                                 pin_memory=True,
                                                 num_workers=0)
                  for x in ['train', 'val']}

    # Loss function and optimizer
    criterion = get_loss(configs['loss'], configs['label_type'])

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=configs['learning_rate'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                           amsgrad=True)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=configs['lr_patience'], verbose=True,
                                  min_lr=6e-5)

    # Auxiliary variables for training process
    epochs_wo_improvement, train_loss, val_loss, best_loss, train_iou, val_iou = 0, [], [], 1e4, [], []
    since = time.time()

    # Training process
    for epoch in range(configs['max_epochs']):

        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, configs['max_epochs']))
        print('-' * 10)

        start = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluation mode

            running_loss, running_iou = 0.0, 0.0

            # Iterate over data
            for samples in dataloader[phase]:

                # Get img_batch and label_batch and put them on GPU if available
                if len(samples) == 2:
                    img_batch, label_batch = samples
                    img_batch, label_batch = img_batch.to(device), label_batch.to(device)
                elif len(samples) == 3:
                    img_batch, border_label_batch, cell_label_batch = samples
                    img_batch = img_batch.to(device)
                    cell_label_batch, border_label_batch = cell_label_batch.to(device), border_label_batch.to(device)
                elif len(samples) == 4:
                    img_batch, border_label_batch, cell_dist_label_batch, cell_label_batch = samples
                    img_batch = img_batch.to(device)
                    cell_label_batch = cell_label_batch.to(device)
                    border_label_batch = border_label_batch.to(device)
                    cell_dist_label_batch = cell_dist_label_batch.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass (track history if only in train)
                with torch.set_grad_enabled(phase == 'train'):

                    if configs['label_type'] in ['boundary', 'border', 'pena']:
                        pred_batch = net(img_batch)
                        loss = criterion(pred_batch, label_batch)
                    elif configs['label_type'] == 'adapted_border':
                        border_pred_batch, cell_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss = 0.0075 * loss_border + loss_cell
                    elif configs['label_type'] == 'distance':
                        border_pred_batch, cell_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss = loss_border + loss_cell
                    elif configs['label_type'] == 'dual_unet':
                        border_pred_batch, cell_dist_pred_batch, cell_pred_batch = net(img_batch)
                        loss_border = criterion['border'](border_pred_batch, border_label_batch)
                        loss_cell_dist = criterion['cell_dist'](cell_dist_pred_batch, cell_dist_label_batch)
                        loss_cell = criterion['cell'](cell_pred_batch, cell_label_batch)
                        loss = 0.01 * loss_border + 0.01 * loss_cell + loss_cell_dist

                    # Backward (optimize only if in training phase)
                    if phase == 'train':

                        loss.backward()
                        optimizer.step()

                    # IoU metric
                    if configs['label_type'] in ['boundary', 'border', 'pena']:
                        iou = iou_pytorch(pred_batch[:, 1, :, :], label_batch == 1, device)
                    elif configs['label_type'] == 'adapted_border':
                        iou_border = iou_pytorch(border_pred_batch[:, 1, :, :], border_label_batch == 1, device)
                        iou_cells = iou_pytorch(cell_pred_batch, cell_label_batch, device)
                        iou = (iou_border + iou_cells) / 2
                    elif configs['label_type'] == 'distance':
                        iou_border = iou_pytorch(border_pred_batch,
                                                 border_label_batch > torch.tensor([0.5],
                                                                                   requires_grad=False).to(device),
                                                 device)
                        iou_cells = iou_pytorch(cell_pred_batch,
                                                cell_label_batch > torch.tensor([0.5],
                                                                                requires_grad=False).to(device),
                                                device)
                        iou = (iou_border + iou_cells) / 2
                    elif configs['label_type'] == 'dual_unet':
                        iou = iou_pytorch(cell_pred_batch, cell_label_batch, device)

                # Statistics
                running_loss += loss.item() * img_batch.size(0)
                running_iou += iou * img_batch.size(0)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_iou = running_iou / len(datasets[phase])

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_iou.append(epoch_iou)
                print('Training loss: {:.4f}, iou: {:.4f}'.format(epoch_loss, epoch_iou))
            else:
                val_loss.append(epoch_loss)
                val_iou.append(epoch_iou)
                print('Validation loss: {:.4f}, iou: {:.4f}'.format(epoch_loss, epoch_iou))

                scheduler.step(epoch_loss)

                if epoch_loss < best_loss:
                    print('Validation loss improved from {:.4f} to {:.4f}. Save model.'.format(best_loss, epoch_loss))
                    best_loss = epoch_loss

                    # The state dict of data parallel (multi GPU) models need to get saved in a way that allows to
                    # load them also on single GPU or CPU
                    if configs['num_gpus'] > 1:
                        torch.save(net.module.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                    else:
                        torch.save(net.state_dict(), str(path_models / (configs['run_name'] + '.pth')))
                    epochs_wo_improvement = 0

                else:
                    print('Validation loss did not improve.')
                    epochs_wo_improvement += 1

        # Epoch training time
        print('Epoch training time: {:.1f}s'.format(time.time() - start))

        # Break training if plateau is reached
        if epochs_wo_improvement == configs['break_condition']:
            print(str(epochs_wo_improvement) + ' epochs without validation loss improvement --> break')
            break

    # Total training time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 20)
    configs['training_time'], configs['trained_epochs'] = time_elapsed, epoch + 1

    # Save loss
    stats = np.transpose(np.array([list(range(1, len(train_loss) + 1)), train_loss, train_iou, val_loss, val_iou]))
    np.savetxt(fname=str(path_models / (configs['run_name'] + '_loss.txt')), X=stats,
               fmt=['%3i', '%2.5f', '%1.4f', '%2.5f', '%1.4f'],
               header='Epoch, training loss, training iou, validation loss, validation iou', delimiter=',')

    # Clear memory
    del net
    gc.collect()

    return None

def train_fn(model, 
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
  train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
  val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


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
              loss = focal_loss(masks_pred, true_masks.squeeze(0), alpha=0.25, gamma = 2, reduction='mean').unsqueeze(0)
              loss += dice_loss(masks_pred, true_masks.squeeze(0), True, k = 0.75)

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

def train_model(model, 
            device,
            training_set,
            validation_set,
            epochs=50,
            n_classes=2,
  
):

  train_loader = DataLoader(training_set, batch_size=8, shuffle=True, num_workers=0)
  valid_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
  # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
  # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
  loss = sm.utils.losses.DiceLoss()
  metrics = [
      sm.utils.metrics.IoU(threshold=0.5),
  ]

  optimizer = torch.optim.Adam([ 
      dict(params=model.parameters(), lr=0.0001),
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
  # train model for 40 epochs

  max_score = 0

  for i in range(0, epochs):
      
      print('\nEpoch: {}'.format(i))
      train_logs = train_epoch.run(train_loader)
      valid_logs = valid_epoch.run(valid_loader)
      
      # do something (save model, change lr, etc.)
         
      if max_score < valid_logs['iou_score']:
          max_score = valid_logs['iou_score']          
          try:
            os.mkdir(RESULTS_PATH)
            logging.info('Created checkpoint directory')
          except OSError:
              pass
          torch.save(model,
                      RESULTS_PATH + f'CP_epoch{i + 1}.pth')
          logging.info(f'Checkpoint {i + 1} saved !')
      if i == 20:
          optimizer.param_groups[0]['lr'] = 1e-5
          print('Decrease decoder learning rate to 1e-5!')

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

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
    

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
  if os.path.exists(MAIN_BOUNDARY_DIR):
    shutil.rmtree(MAIN_BOUNDARY_DIR)
  os.mkdir(MAIN_BOUNDARY_DIR)

  add_class_to_image_name(DATASET_NAMES, TRAIN_IMG_DIRS, MAIN_IMAGE_DIR)
  add_class_to_image_name(DATASET_NAMES, TRAIN_MASK_DIRS, MAIN_MASK_DIR)
  #threshold_masks(DATASET_NAMES, MAIN_MASK_DIR)
  create_boundary_masks(MAIN_MASK_DIR, MAIN_BOUNDARY_DIR)

  train_img_dir, val_img_dir, train_mask_dir, val_mask_dir, train_boundary_dir, val_boundary_dir =   split_train_val_set(MAIN_IMAGE_DIR, MAIN_MASK_DIR, MAIN_BOUNDARY_DIR, TRAIN_VAL_SPLIT)

  generate_augmented_images(train_img_dir, train_mask_dir, train_boundary_dir, AUGMENTATION_PER_IMAGE, get_training_augmentation())
  train_ds = get_loaders(
      train_img_dir=train_img_dir,
      train_mask_dir=train_mask_dir,
      train_boundary_dir=train_boundary_dir,
      batch_size=BATCH_SIZE,
      max_size=IMAGE_SIZE,
      train_transform=get_training_augmentation(),
      train_classes=DATASET_NAMES,
      train_preprocessing=get_preprocessing(preprocessing_fn)
      )
  val_ds = get_loaders(
    train_img_dir=val_img_dir,
    train_mask_dir=val_mask_dir,
    train_boundary_dir=val_boundary_dir,
    batch_size=BATCH_SIZE,
    max_size=IMAGE_SIZE,
    train_transform=get_validation_augmentation(),
    train_classes=DATASET_NAMES,
    train_preprocessing=get_preprocessing(preprocessing_fn)
    )


  # model = sm.EfficientUnetPlusPlus(BACKBONE, encoder_weights=ENCODER_WEIGHTS, classes=NUM_CLASSES, activation=ACTIVATION)
  # # Distribute training over GPUs
  # model = nn.DataParallel(model)



  # train_fn(model=model, 
  #           device=device,
  #           training_set=train_ds,
  #           validation_set=val_ds,
  #           dir_checkpoint=RESULTS_PATH,
  #           epochs=EPOCHS,
  #           batch_size=BATCH_SIZE,
  #           lr=LR,
  #           save_cp=True,
  #           img_scale=1,
  #           n_classes=NUM_CLASSES,
  #           n_channels=3,
  #           augmentation_ratio = AUGMENTATION_PER_IMAGE)

  # train_model(model=model, 
  #           device=device,
  #           training_set=train_ds,
  #           validation_set=val_ds,
  #           epochs=EPOCHS,
  #           n_classes=NUM_CLASSES
  #           )


if __name__ == "__main__":
  main()

