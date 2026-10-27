import argparse
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from data.brats19_datamodule import BRATSDataModule
from models.hdcl_brats19_lightning import HDCL


def parse_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--lambda_disentangle_shared', type=float, default=1)
    parser.add_argument('--lambda_disentangle_ehr', type=float, default=1)
    parser.add_argument('--lambda_disentangle_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_ehr', type=float, default=1)
    parser.add_argument('--lambda_pred_cxr', type=float, default=1)
    parser.add_argument('--lambda_pred_shared', type=float, default=1)
    parser.add_argument('--semi_rate', type=float, default=0.1) 

    parser.add_argument('--lr', type=float, default=0.01) #0.0001 / 0.01-SGD
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=10)

    parser.add_argument("--weight_std", action='store_true', default=True)
    parser.add_argument("--norm_cfg", type=str, default='IN')  # normalization
    parser.add_argument("--activation_cfg", type=str, default='LeakyReLU')  # activation
    parser.add_argument('--patch_shape', default=[96,96,96], type=list, help='input shape, default to 96')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_classes', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', help='train or val')  
    parser.add_argument('--data_root', default='/users/fxzhou/fastscratch/project_data/MICCAI_BraTS_2019_Data_Training', type=str, help='root path to images')
    parser.add_argument('--json_file', default='dataset_braTS19.json', type=str, help='json file name')
    parser.add_argument('--save_path', type=str, default='logs/test')

    return parser.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    dm = BRATSDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args,
    )
    dm.setup()


    # initialize lightning module
    model = HDCL(args)

    # setup loggers and checkpoints
    # version_name = f"train_{args.semi_rate:.2f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=args.save_path,
        name="",
        version=""
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",          
        mode="max",                 
        save_top_k=1,  
        save_last=True,        
        filename="best-epoch{epoch:03d}-dice{val_dice:.4f}",
        auto_insert_metric_name=False,
    )


    # ckpt_periodic = ModelCheckpoint(
    #     every_n_epochs=5,
    #     save_top_k=-1,
    #     filename="epoch{epoch:03d}",
    # )

    ckpt_best = ModelCheckpoint(
        monitor="val_dice",
        mode="max",
        save_top_k=1,
        save_last=True,        
        filename="best-epoch{epoch:03d}-dice{val_dice:.4f}",
    )


    # initialize Lightning's trainer.
    trainer = pl.Trainer(
        # num_sanity_val_steps =0,
        accelerator="gpu",
        devices=1,                 
        strategy="ddp",             
        max_epochs=1000,
        precision=16,
        logger=tb_logger,
        check_val_every_n_epoch=5,
        # multiple_trainloader_mode="min_size",  #supervised
        multiple_trainloader_mode="max_size_cycle",
        callbacks=[checkpoint_callback],
        # callbacks=[ckpt_periodic, ckpt_best],
    )

    ckpt_path = os.path.join(args.save_path, 'checkpoints/last.ckpt')
    if os.path.exists(ckpt_path):
        print("load from--", ckpt_path)

        model = HDCL.load_from_checkpoint(ckpt_path, args=args)

        if args.mode=='train':
            trainer.fit(
                model,
                dm,
                ckpt_path=ckpt_path    
            )

    if args.mode=='train':
            trainer.fit(
                model,
                dm,
            )

    if args.mode=='val':
        print("Start validation...")
        # load best checkpoint
        # ckpt_path = "./logs/brats19/0.1/unet_heri_crossfusion_baseline_0.1loss_noisecons_0.1loss_predcons_0.1loss_fused2shared_spec_cons4/checkpoints/best-epoch224-dice0.8515.ckpt"
        # ckpt_path = "./checkpoints/brats19/0.1/best-epoch224-dice0.8515.ckpt"

        # load model
        model = HDCL.load_from_checkpoint(
                ckpt_path,
                args=args,
                strict=False,
            )
        model.eval()

        # validate only
        val_results = trainer.validate(model, datamodule=dm)
        print("Validation results:", val_results)


if __name__ == "__main__":
    main()
