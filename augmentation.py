import numpy as np
import albumentations as albu
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from functional import (circle_rotate,
    generate_cutmix_image,
    randomize_custom_aug_seeds
)


###############################################################################
# From here we create the list of transforms that we will consider for augmentation
###############################################################################



def rand_augment(N):
  transforms = [
      albu.NoOp(p=1),
      # albu.Equalize(p=1),
      # albu.Solarize(p=1),
      # albu.ColorJitter(p=1),
      # albu.Posterize(p=1),
      # albu.RandomBrightnessContrast(p=1),
      # albu.Sharpen(p=1),
      # albu.InvertImg(p=1),
      # albu.GaussNoise(p=1),
      # albu.Rotate(p=1),
      # albu.RandomGridShuffle (grid=(3, 3), p=1),
      # albu.ElasticTransform(p=1, alpha=200, sigma=12, alpha_affine=9),      
      # #Random Local Rotate
      # albu.Lambda(image=circle_rotate,
      #             mask=circle_rotate,
      #             name='circle_rotate',                 
      #             p=1.0),
      #Cut Mix
      albu.Lambda(image=generate_cutmix_image,
                  mask=generate_cutmix_image,
                  name='generate_cutmix_image',                 
                  p=1.0),    
    ]
  randomize_custom_aug_seeds()
  sampled_ops = np.random.choice(transforms, N)
  return albu.Compose(sampled_ops)

def get_training_augmentation():
    train_transform = [
        albu.VerticalFlip(p=0.5),              
        albu.RandomRotate90(p=1),
        albu.HorizontalFlip(p=0.5),
        albu.Transpose(p=0.5),
        #albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        albu.ElasticTransform(p=0.5, alpha=10, sigma=3, alpha_affine=3),
        albu.GridDistortion(p=0.5),
        albu.InvertImg(p=0.5),
        albu.RandomBrightnessContrast(p=0.5),
        albu.Sharpen(p=0.5),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose    
    """    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
