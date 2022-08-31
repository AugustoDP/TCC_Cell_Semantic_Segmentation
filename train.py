import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
from os import path
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split
from utils import (
  save_checkpoint,
  load_checkpoint,
  get_loaders,
)

TRAIN_IMG_DIRS = '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01'
TRAIN_MASK_DIRS = '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST/SEG'
BATCH_SIZE = 32
EPOCHS = 75
LOAD_MODEL = False
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
OPTIMIZER = 'Adam'
LOSS = sm.losses.binary_crossentropy
METRICS = sm.metrics.iou_score
BACKBONE = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IMAGES_TO_GENERATE = 500
VALIDATION_SPLIT = 0.2
TEST_IMG = '/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/02/t000.tif'
RESULTS_PATH ="/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/Results" # path to store model results


def train_fn(loader, model):
  preprocess_input = sm.get_preprocessing(BACKBONE)

  X, Y = loader.__get_img_mask_list__(height=IMAGE_HEIGHT, width=IMAGE_WIDTH)
  Y = np.expand_dims(Y, axis=3) #May not be necessary.. leftover from previous code 

  
  x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=VALIDATION_SPLIT, random_state=42)
  # preprocess input
  x_train = preprocess_input(x_train)
  x_val = preprocess_input(x_val)

  print(model.summary())
  history=model.fit(x_train, 
            y_train,
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_val, y_val))

  #accuracy = model.evaluate(x_val, y_val)
  #plot the training and validation accuracy and loss at each epoch
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs = range(1, len(loss) + 1)
  plt.plot(epochs, loss, 'y', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
  return model

def predict(model, image_path):
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)       
  image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  prediction = model.predict(image)
  #View and Save segmented image
  prediction_image = prediction.reshape([256,256,1])
  new_img = cv2.cvtColor(prediction_image, cv2.COLOR_BGR2RGB)
  plt.imshow(new_img, cmap='gray')
  prediction_image_name = 'test_' + os.path.basename(image_path) 
  output_path = os.path.join(RESULTS_PATH, prediction_image_name)
  plt.imsave(output_path, new_img, cmap='gray')


def main():
  train_transform = A.Compose([
      A.VerticalFlip(p=0.5),              
      A.RandomRotate90(p=1),
      A.HorizontalFlip(p=0.5),
      A.Transpose(p=0.5),
      #A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
      A.ElasticTransform(p=0.5, alpha=10, sigma=3, alpha_affine=3),
      A.GridDistortion(p=0.5),
      A.InvertImg(p=0.5),
      A.RandomBrightnessContrast(p=0.5),
      A.Sharpen(p=0.5),
      #A.Normalize(
      #  mean=[0.0, 0.0, 0.0],
      #  std=[1.0, 1.0, 1.0],
      #  max_pixel_value=255.0,
      #),
      #A.RandomCrop(height=50, width=50, p=0.5)
      ]
  )
  if os.path.exists(RESULTS_PATH) == False:
    os.mkdir(RESULTS_PATH)
  train_ds = get_loaders(
      TRAIN_IMG_DIRS,
      TRAIN_MASK_DIRS,
      BATCH_SIZE,
      train_transform,
      )
  train_ds.__apply__(IMAGES_TO_GENERATE)
  train_ds.__read_augmented__()
  model = sm.Unet(BACKBONE, encoder_weights=ENCODER_WEIGHTS, classes=1)
  model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS])
  model = train_fn(train_ds, model)
  save_checkpoint(model, RESULTS_PATH)
  predict(model, TEST_IMG)


if __name__ == "__main__":
  main()

