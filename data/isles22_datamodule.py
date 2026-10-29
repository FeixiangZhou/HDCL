import numpy as np
import pytorch_lightning as pl
import torch
from monai import transforms
from monai.data import DataLoader, Dataset
from dataset.transforms import isles_train_transform, isles_val_transform

import factorizer as ft

# def get_train_transform(args):
#     train_transform = [
#         transforms.LoadImaged(["image", "label"]),
#         transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
#         transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2

#         transforms.CropForegroundd(["image", "label"], source_key="image"),
#         transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
#         transforms.Spacingd(
#             ["image", "label"],
#             pixdim=(2.0, 2.0, 2.0),
#             mode=("bilinear", "bilinear"),
#         ),
#         transforms.RandSpatialCropd(["image", "label"], roi_size=args.patch_shape, random_size=False),
#         transforms.RandAffined(
#             ["image", "label"],
#             prob=0.15,
#             spatial_size=args.patch_shape,
#             rotate_range=[30 * np.pi / 180] * 3,
#             scale_range=[0.3] * 3,
#             mode=("bilinear", "bilinear"),
#         ),
#         transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=0),
#         transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=1),
#         transforms.RandFlipd(["image", "label"], prob=0.5, spatial_axis=2),
#         transforms.RandGaussianNoised("image", prob=0.15, std=0.1),
#         transforms.RandGaussianSmoothd(
#             "image",
#             prob=0.15,
#             sigma_x=(0.5, 1.5),
#             sigma_y=(0.5, 1.5),
#             sigma_z=(0.5, 1.5),
#         ),
#         transforms.RandScaleIntensityd("image", prob=0.15, factors=0.3),
#         transforms.RandShiftIntensityd("image", prob=0.15, offsets=0.1),
#         transforms.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
#         transforms.AsDiscreted("label", threshold=0.5),
#         transforms.ToTensord(["image", "label"]),
#     ]
#     train_transform = transforms.Compose(train_transform)
#     return train_transform


# def get_val_transform():
#     val_transform = [
#         transforms.LoadImaged(["image", "label"]),
#         transforms.EnsureChannelFirstd(keys=["image", "label"], allow_missing_keys=True),
#         transforms.Lambdad(keys="image", func=lambda x: x[:2, ...]),  # use C=2
#         transforms.NormalizeIntensityd("image", nonzero=True, channel_wise=True),
#         transforms.ToTensord(["image", "label"], allow_missing_keys=True),
#     ]
#     val_transform = transforms.Compose(val_transform)
#     return val_transform


class ISLESDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_properties,
        args,
        fold=0,
        batch_size=2,
        num_workers=None,
        **kwargs,
    ):
        super().__init__()
        
        self.args = args
        self.data_properties = ft.load_properties(data_properties)
        self.fold = fold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = kwargs

        self.train_transform = isles_train_transform(spatial_size=tuple(self.args.patch_shape))
        self.val_transform = isles_val_transform()
        self.test_transform = isles_val_transform()

        self.train_set = self.val_set = self.test_set = None
    
    def setup(self, stage=None):
        if stage in ("fit", "validate", None):
            # make training set
            training_data = []
            val_data = []
            for sample in self.data_properties["training"]:
                if sample["fold"] == self.fold:
                    val_data.append(sample)
                else:
                    training_data.append(sample)

            training_data_labeled = training_data[0: int(len(training_data) * self.args.semi_rate)]
            training_data_unlabeled = training_data[int(len(training_data) * self.args.semi_rate):]
           
            # training_data_unlabeled = training_data_labeled # for fully supervised learning


            self.train_set = Dataset(
                training_data_labeled,
                transform=self.train_transform,
                **self.dataset_kwargs,
            )
            self.train_set_unlabeled = Dataset(
                            training_data_unlabeled,
                            transform=self.train_transform,
                            **self.dataset_kwargs,
            )
            
            self.val_set = Dataset(
                val_data,
                transform=self.val_transform,
                **self.dataset_kwargs,
            )

        if stage in ("test", "predict", None):
            # make test set
            self.test_set = Dataset(
                self.data_properties["test"],
                transform=self.test_transform,
                **self.dataset_kwargs,
            )

    def train_dataloader(self):
        train_loader_labeled = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        train_loader_unlabeled = DataLoader(
            self.train_set_unlabeled,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # print("train_loader_labeled--------",len(train_loader_labeled)) #10
        # print("train_loader_unlabeled--------",len(train_loader_unlabeled)) #90

        return {
        "labeled": train_loader_labeled,
        "unlabeled": train_loader_unlabeled,
    }

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=1,
            num_workers=self.num_workers,
        )
        return test_loader
