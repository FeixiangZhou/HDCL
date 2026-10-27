import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from monai.data import DataLoader, Dataset

from dataset.transforms import brats_train_transform, brats_val_transform

class BRATSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        batch_size=2,
        num_workers=None,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.args = args
        self.dataset_kwargs = kwargs

        self.train_transform = brats_train_transform(spatial_size=tuple(self.args.patch_shape))
        self.val_transform = brats_val_transform()
        self.test_transform = brats_val_transform()


        # self.train_transform = custom_transform(patch_shape=self.args.patch_shape, mode='train')
        # self.val_transform = custom_transform(patch_shape=self.args.patch_shape, mode='val')
        # self.test_transform = custom_transform(patch_shape=self.args.patch_shape, mode='val')

        self.train_set = self.val_set = self.test_set = None

    def format_input(self, data_root: Path, 
                 filename_list: list) -> list:
        fullname_list = [
            {
                'image': [data_root / y for y in x['image']],
                'label': data_root / x['label'],
                'id': x['id']
            } for x in filename_list
        ]
        return fullname_list
    

    def setup(self, stage=None):
        data_root = Path(self.args.data_root)
        with open(os.path.join(data_root, self.args.json_file), 'r') as fr:
            data_list = json.load(fr)

        training_data = self.format_input(data_root, data_list["training"])
        val_data = self.format_input(data_root, data_list["validation"])
        test_data = self.format_input(data_root, data_list["test"])
        print("Train: %d \nValidation: %d \nTest: %d" % (len(training_data), len(val_data), len(test_data)))
        

        training_data_labeled = training_data[0: int(len(training_data) * self.args.semi_rate)]
        training_data_unlabeled = training_data[int(len(training_data) * self.args.semi_rate):]
       
        # training_data_unlabeled = training_data_labeled # for fully


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
            test_data,
            transform=self.val_transform,
            **self.dataset_kwargs,
        )

        # make test set
        self.test_set = Dataset(
            test_data,
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

        print("train_loader_labeled--------",len(train_loader_labeled)) 
        print("train_loader_unlabeled--------",len(train_loader_unlabeled)) 

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
