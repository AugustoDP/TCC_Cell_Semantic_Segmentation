import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
from os import path
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from utils import (
  save_checkpoint,
  load_checkpoint,
  get_loaders,
)

TRAIN_IMG_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01']
TRAIN_MASK_DIRS = ['/content/TCC_Cell_Semantic_Segmentation/DIC-C2DH-HeLa/01_ST/SEG']
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


def train_fn(loader, model, optimizer, loss_fn, scaler):
  pass


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
      A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
      ),
      #A.RandomCrop(height=50, width=50, p=0.5)
      ]
  )


if __name__ == "__main__":
  main()