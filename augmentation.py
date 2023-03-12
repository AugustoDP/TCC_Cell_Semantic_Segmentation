import numpy as np
import albumentations as albu

from PIL import Image, ImageDraw, ImageFilter, ImageFont

def circle_rotate(image, **kwargs):
    radius=np.random.randint(60, 121)
    x=np.random.randint(radius, 257-radius)
    y=np.random.randint(radius, 257-radius)
    degree=np.random.randint(-90, 91)
    box = (x-radius,y-radius,x+radius+1,y+radius+1)
    img = Image.fromarray(image)
    background = Image.new("RGBA", img.size, (0,0,0,0))
    mask = Image.new("RGBA", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse(box, fill='green', outline=None)
    new_img = Image.composite(img, background, mask)
    crop = new_img.crop(box=box)
    crop = crop.rotate(degree)
    background.paste(crop, (x-radius, y-radius))
    new_img = Image.composite(background, img, mask)
    return np.array(new_img)

transforms = [
      albu.NoOp(p=1),
      albu.Equalize(p=1),
      albu.Rotate(p=1),
      albu.Solarize(p=1),
      albu.ColorJitter(p=1),
      albu.Posterize(p=1),
      albu.RandomBrightnessContrast(p=1),
      albu.Sharpen(p=1),
      albu.RandomGridShuffle (grid=(3, 3), p=1),
      albu.InvertImg(p=1),
      albu.CoarseDropout(max_holes=1, max_height=75, max_width=75, fill_value=0, mask_fill_value=0, p=1),
      albu.ElasticTransform(p=1, alpha=150, sigma=6, alpha_affine=6),
      #albu.MultiplicativeNoise(multiplier=(0.9, 1.6), per_channel=False, elementwise=False, p=1),
      albu.GaussNoise(var_limit=(25.0, 75.0), mean=128, per_channel=True, p=1),
      #albu.CropNonEmptyMaskIfExists(height, width, ignore_values=None, ignore_channels=None, p=1.0)
      # Affine operations for translate x and y, shear x and y
      albu.Affine(translate_percent={"x": (0,0.5)}, 
                  interpolation=1,
                  mask_interpolation=0, 
                  cval=0, 
                  cval_mask=0, 
                  mode=0,                 
                  fit_output=False, 
                  keep_ratio=False,
                  p=1),
      albu.Affine(translate_percent={"y": (0,0.5)}, 
                  interpolation=1,
                  mask_interpolation=0, 
                  cval=0, 
                  cval_mask=0, 
                  mode=0,                 
                  fit_output=False, 
                  keep_ratio=False,
                  p=1),
      albu.Affine(shear={"x": (0,0.5)},
                  interpolation=1,
                  mask_interpolation=0, 
                  cval=0, 
                  cval_mask=0, 
                  mode=0,                 
                  fit_output=False, 
                  keep_ratio=False,
                  p=1),
      albu.Affine(shear={"y": (0,0.5)},
                  interpolation=1,
                  mask_interpolation=0, 
                  cval=0, 
                  cval_mask=0, 
                  mode=0,                 
                  fit_output=False, 
                  keep_ratio=False,
                  p=1),
      #Random Local Rotate
      albu.Lambda(image=circle_rotate,
                name='circle_rotate',                 
                p=1.0)
    ]


def rand_augment(N):
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
        # albu.Normalize(
        #   mean=[0.0, 0.0, 0.0],
        #   std=[1.0, 1.0, 1.0],
        #   max_pixel_value=255.0,
        # ),
        #albu.RandomCrop(height=50, width=50, p=0.5)


        # albu.HorizontalFlip(p=0.5),

        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),

        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
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
