

import albumentations as albu

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
    return albu.Compose(train_transform, additional_targets={'boundary':'mask'})

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform, additional_targets={'boundary':'mask'})

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
