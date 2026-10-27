import monai.transforms as transforms
from monai.transforms.utility.dictionary import ConvertToMultiChannelBasedOnBratsClassesD
from .mix_transform import RandomChannelCutmixd
from monai.transforms import MapTransform
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np

class ConvertToMultiChannelStroke:
    def __call__(self, label):
        # print("ConvertToMultiChannelStroke--", torch.unique(label))
        # label: [D,H,W] or [1,D,H,W]
        if label.dim() == 3:
            label = label.unsqueeze(0)

        label = label.long()

        ch = (label == 1).float()   # binary lesion
        return ch.contiguous()      # shape: [1,D,H,W]
    

class ConvertToMultiChannelStrokeD(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelStroke()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class ConvertToMultiChannelBrats23:
    def __call__(self, label):
        # print("ConvertToMultiChannelBrats23--------------------", torch.unique(label))
        ch0 = (label == 3).float() # tumor enhancing
        ch1 = (label >=1).float()   # whole   tumor
        ch3 = torch.logical_or(label == 1, label == 3).float()# tumor core

        return torch.cat([ch0, ch1, ch3], dim=0)


class ConvertToMultiChannelBrats23D(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBrats23()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ConvertToMultiChannelBrats19:
    def __call__(self, label):
        if label.dim() == 3:
            label = label.unsqueeze(0)
        # print("ConvertToMultiChannelBrats23--------------------", torch.unique(label))
        # ch0 = (label == 4).float() # tumor enhancing
        ch1 = (label >=1).float()   # whole   tumor
        # ch3 = torch.logical_or(label == 1, label == 4).float()# tumor core

        # return torch.cat([ch0, ch1, ch3], dim=0)
        return ch1.contiguous() #only WT


class ConvertToMultiChannelBrats19D(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBrats19()

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

def custom_transform(patch_shape=96, mode='train',enable_channel_cutmix=False, pair_aug=False, in_channels=4): 
    if mode == 'train':
        if enable_channel_cutmix:
            train_transform = transforms.Compose([
                transforms.LoadImaged(keys=['image', 'label', 'template']),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                # transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2
                transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
                transforms.EnsureTyped(keys=["label"], dtype=torch.long),
                transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True),
                transforms.Orientationd(keys=['image', 'label', 'template'], axcodes='RAS'),
                transforms.Spacingd(keys=['image', 'label', 'template'], pixdim=(1.0, 1.0, 1.0), mode=['bilinear', 'nearest', 'bilinear']),
                transforms.CropForegroundd(keys=['image', 'template', 'label'], source_key='image', margin=1),
                RandomChannelCutmixd(keys=['image'], num_mix=3, pair_aug=pair_aug, max_num_channel=in_channels),
                transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=patch_shape, random_size=False),  
                # transforms.RandCropByPosNegLabeld(keys=["image", "label"],label_key="label",spatial_size=patch_shape, pos=1, neg=1, num_samples=4),
                transforms.SpatialPadd(keys=['image', 'label'], spatial_size=patch_shape, mode='constant'),
                transforms.RandFlipd(keys=['image', 'label'], spatial_axis=(0, 1, 2), prob=0.5),
                transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
                transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
                # ConvertToMultiChannelBrats23D(keys='label'), # region-based segmentation   
                # ConvertToMultiChannelStrokeD(keys='label'), # binary segmentation


            ])
        else:
            train_transform = transforms.Compose([
                transforms.LoadImaged(keys=['image', 'label']),
                transforms.EnsureChannelFirstd(keys=["image", "label"]),
                # transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2
                # image → float
                transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
                # label → long（或 float，但必须是整数）
                transforms.EnsureTyped(keys=["label"], dtype=torch.long),
                transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True),
                transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                transforms.Spacingd(keys=['image', 'label'], pixdim=(2.0, 2.0, 2.0), mode=['bilinear', 'nearest']),
                transforms.CropForegroundd(keys=['image', 'label'], source_key='image', margin=1),
                transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=patch_shape, random_size=False),  
                # transforms.RandCropByPosNegLabeld(keys=["image", "label"],label_key="label",spatial_size=patch_shape, pos=1, neg=1, num_samples=4),
                transforms.SpatialPadd(keys=['image', 'label'], spatial_size=patch_shape, mode='constant'),
                transforms.RandFlipd(keys=['image', 'label'], spatial_axis=(0, 1, 2), prob=0.5),
                transforms.RandShiftIntensityd(keys=['image'], offsets=0.1, prob=1.0),
                transforms.RandScaleIntensityd(keys=['image'], factors=0.1, prob=1.0),
                # ConvertToMultiChannelBrats23D(keys='label'), # region-based segmentation
                # ConvertToMultiChannelStrokeD(keys='label'), # binary segmentation

            ])
        return train_transform
    
    val_transform = transforms.Compose([
            transforms.LoadImaged(keys=['image', 'label']),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            # transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]), 
            transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
            transforms.EnsureTyped(keys=["label"], dtype=torch.long),
            transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True),
            transforms.Orientationd(keys=['image', 'label'], axcodes='RAS'),
            transforms.Spacingd(keys=['image', 'label'], pixdim=(2.0, 2.0, 2.0), mode=['bilinear', 'nearest']),
            # transforms.CropForegroundd(keys=['image', 'label'], source_key='image', margin=1),
            # ConvertToMultiChannelBrats23D(keys='label'), # region-based segmentation
            # ConvertToMultiChannelStrokeD(keys='label'), # binary segmentation

        ])
    return val_transform



def isles_train_transform(spatial_size=(96, 96, 64), version="brats19"):
    train_transform = [
        transforms.LoadImaged(["image", "label"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2

        transforms.CropForegroundd(["image", "label"], source_key="image"),
        transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
        transforms.Spacingd(
            ["image", "label"],
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear", "nearest"),
        ),
        transforms.RandSpatialCropd(["image", "label"], roi_size=spatial_size, random_size=False),
        transforms.RandAffined(
            ["image", "label"],
            prob=0.15,
            spatial_size=spatial_size,
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "nearest"),
        ),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandGaussianNoised("image", prob=0.15, std=0.1),
        transforms.RandGaussianSmoothd(
            "image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        transforms.RandScaleIntensityd("image", prob=0.15, factors=0.3),
        transforms.RandShiftIntensityd("image", prob=0.15, offsets=0.1),
        transforms.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform


def isles_val_transform():
    val_transform = [
        transforms.LoadImaged(["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2
        transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform



def brats_train_transform(spatial_size=(128, 128, 128), version="brats19"):
    train_transform = [
        transforms.LoadImaged(["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2
        # transforms.Lambdad(keys="image", func=lambda x: x[0:1, ...]),  # use t1ce
        # transforms.Lambdad(keys="image", func=lambda x: x[1:2, ...]),  # use t2

        ConvertToMultiChannelBrats19D(keys='label'),
        transforms.CropForegroundd(["image", "label"], source_key="image"),
        transforms.NormalizeIntensityd(
            "image", nonzero=True, channel_wise=True
        ),
        transforms.RandSpatialCropd(
            ["image", "label"], roi_size=spatial_size, random_size=False
        ),
        transforms.RandAffined(
            ["image", "label"],
            prob=0.15,
            spatial_size=spatial_size,
            rotate_range=[30 * np.pi / 180] * 3,
            scale_range=[0.3] * 3,
            mode=("bilinear", "bilinear"),
        ),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandGaussianNoised("image", prob=0.15, std=0.1),
        transforms.RandGaussianSmoothd(
            "image",
            prob=0.15,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),
        transforms.RandScaleIntensityd("image", prob=0.15, factors=0.3),
        transforms.RandShiftIntensityd("image", prob=0.15, offsets=0.1),
        transforms.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
        transforms.AsDiscreted("label", threshold=0.5),
        transforms.ToTensord(["image", "label"]),
    ]
    train_transform = transforms.Compose(train_transform)
    return train_transform


def brats_val_transform(version="brats19"):
    val_transform = [
        transforms.LoadImaged(["image", "label"], allow_missing_keys=True),
        transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
        transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2
        # transforms.Lambdad(keys="image", func=lambda x: x[0:1, ...]),  # use t1ce
        # transforms.Lambdad(keys="image", func=lambda x: x[1:2, ...]),  # use t2


        ConvertToMultiChannelBrats19D(keys='label'),
        transforms.NormalizeIntensityd(
            "image", nonzero=True, channel_wise=True
        ),
        transforms.ToTensord(["image", "label"], allow_missing_keys=True),
    ]
    val_transform = transforms.Compose(val_transform)
    return val_transform
