import copy
import csv
import os

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from monai.data import MetaTensor
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, MeanIoU, SurfaceDistanceMetric
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.unet_shaspec_aug_missing_heri_crossfusion import UNet

from utils.losses import (
    CosineOrthogonalLoss,
    MMSelfDistillationLoss1C,
    PredictionConsistencyLoss,
    ROIWeightedConsistencyLoss,
    SharedAlignmentL1Loss,
    sigmoid_rampup,
)

class HDCL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.val_outputs = []   
        self.val_preds = []
        self.consistency = 0.1
        self.consistency_rampup = 100
 

        self.net = UNet(
        in_channels=1,
        out_channels=1,
        spatial_dims=3,
        encoder_depth=[1, 1, 1, 1, 1],
        encoder_width=[32, 64, 128, 256, 512],
        strides=[1, 2, 2, 2, 2],
        decoder_depth=[1, 1, 1, 1],
        num_deep_supr=1,  
    )


        self.loss = DiceCELoss(sigmoid=True, squared_pred=True)
        # self.deeploss = DeepSuprLoss(loss= self.loss, sigmoid=True, squared_pred=True)
        self.cosine_orthogonal_loss = CosineOrthogonalLoss()
        self.shared_alignment_loss = SharedAlignmentL1Loss()
        self.roi_weighted_consistency_loss = ROIWeightedConsistencyLoss()
        self.prediction_consistency_loss = PredictionConsistencyLoss()  
        self.noise_consistency = MMSelfDistillationLoss1C()


        self.dice_metric = DiceMetric(reduction="mean_batch", get_not_nans=True)
        self.hd95_metric = HausdorffDistanceMetric(percentile=95, reduction="mean_batch",include_background=False, )
        self.asd_metric = SurfaceDistanceMetric(reduction="mean_batch", include_background=False,)
        self.iou_metric = MeanIoU(include_background=False, reduction="mean",)

        self.max_epochs = 500
        self.inferer = SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5)
        # self.inferer = SlidingWindowInferer(roi_size=(128, 128, 128), sw_batch_size=1, overlap=0.5)

        self.post = lambda x: (x >= 0).to(torch.uint8)

        self.key = "image"
        self.meta_key = "image_meta_dict"   
        self.spacing = torch.tensor([1., 1., 1.])
    
    def get_current_consistency_weight(self, epoch):
        return self.consistency * sigmoid_rampup(
            epoch, self.consistency_rampup
        )


    def transform(self, batch):
        self.orig_spacing = batch[self.meta_key]["pixdim"][0][1:4].to("cpu")
        # self.orig_spacing = batch["image"].meta["pixdim"][0][1:4].to("cpu")
        
        if torch.allclose(self.orig_spacing, self.spacing):
            out = batch
        else:
            # print("orig spacing:", self.orig_spacing)
            # print("target spacing:", self.spacing)
            out = copy.deepcopy(batch)
            self.orig_size = batch[self.meta_key]["dim"][0][1:4].to("cpu")
            # self.orig_size = batch["image"].meta["dim"][0][1:4].to("cpu")
            self.new_size = self.orig_size * self.orig_spacing / self.spacing
            self.new_size = self.new_size.round().to(device="cpu", dtype=torch.int)
            # print("orig size:", self.orig_size)
            # print("new size:", self.new_size)
            new_size = tuple(int(x) for x in self.new_size.tolist())

            out[self.key] = F.interpolate(
                batch[self.key],
                size=new_size,
                mode="trilinear"
            )

        return out

    def inverse_transform(self, batch):
        if torch.allclose(self.orig_spacing, self.spacing):
            out = batch
        else:
            orig_size = tuple(int(x) for x in self.orig_size.tolist())
            out = copy.deepcopy(batch)
            out[self.key] = F.interpolate(
                batch[self.key],
                size=orig_size,
                mode="trilinear"
            )
        return out


    def num_parameters(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def forward(self, x):
        out= self.net(x) 
        return out
    
    def infer_forward(self, x):
        out  = self.forward(x)
        return out["pred_shared"][0] 


    def configure_optimizers(self):
        # optimizer = AdamW(self.net.parameters(), lr=self.args.lr, weight_decay=0.01) #lr=0.0001
        optimizer = SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0001)  #lr=0.01
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs) 
        return [optimizer], [scheduler]
        # return [optimizer]


    def predict_step(self, batch, batch_idx):
        # print("predict_step x ---", batch["image"].shape) #[2, 3, 64, 64, 64]
        # print("predict_step y ---", batch["label"].shape) #[2, 1, 64, 64, 64]
        # print("predict_step batch keys: ", batch.keys())  #dict_keys(['id', 'fold', 'image', 'label')
        # print("predict_step batch keys: ", batch["image"].meta)  
        # print("image_meta_dict batch keys: ", batch["image"].meta["pixdim"][0][1:4]) # get spacing [2., 2., 2.]
        
        # batch = self.transform(batch)
        print("predict_step x ---", batch["image"].shape) #[1, 2, 240, 240, 155]

        batch["image"] = self.inferer(batch["image"], self.infer_forward)
        # batch = self.inverse_transform(batch)
        y_hat = batch["image"]
        print("predict_step y_hat shape: ", y_hat.shape) #[1, 1, 240, 240, 155]  
        y_hat = self.post(y_hat)
        # print("post y_hat: ", batch['id'], y_hat.max().item(), y_hat.min().item(), y_hat.float().mean().item())

        return y_hat

    def training_step(self, batch, batch_idx):
        batch_labeled = batch["labeled"]
        batch_unlabeled = batch["unlabeled"]
        # print("training_step batch keys: ", batch.keys()) # dict_keys(['id', 'fold', 'image', 'label', 'image_meta_dict', 'label_meta_dict', 'foreground_start_coord', 'foreground_end_coord'])
        x, y = batch_labeled["image"], batch_labeled["label"]
        # print("training_step unique y---", torch.unique(y))
        print("training_step x ---", x.shape) # [2, 2, 64, 64, 64]
        print("training_step y ---", y.shape) # [2, 1, 64, 64, 64]

        x_unlabeled, y_unlabeled = batch_unlabeled["image"], batch_unlabeled["label"]


        # =======remove MetaTensor =======
        if isinstance(x, MetaTensor):
            x = x.as_tensor()
        if isinstance(y, MetaTensor):
            y = y.as_tensor()

        if isinstance(x_unlabeled, MetaTensor):
            x_unlabeled = x_unlabeled.as_tensor()
        if isinstance(y_unlabeled, MetaTensor):
            y_unlabeled = y_unlabeled.as_tensor()
        # ====================================

        out = self.forward(x)
        y_hat_shared = out["pred_shared"]
        y_hat_shared_aug = out["pred_shared_aug"]


        y_hat_shared_m1 = out["pred_shared_m1"]
        y_hat_shared_m2 = out["pred_shared_m2"]

        y_hat_spec_m1 = out["pred_spec_m1"]
        y_hat_spec_m2 = out["pred_spec_m2"]

        shared_m1_ft_final = out["shared_m1_ft"]    
        shared_m2_ft_final = out["shared_m2_ft"]

        spec_m1_ft_final = out["spec_m1_ft"]
        spec_m2_ft_final = out["spec_m2_ft"]


        out_un = self.forward(x_unlabeled)

        y_hat_shared_un = out_un["pred_shared"]  # fused prediction for unlabeled data
        y_hat_shared_un_aug = out_un["pred_shared_aug"]

        y_hat_shared_m1_un = out_un["pred_shared_m1"]
        y_hat_shared_m2_un = out_un["pred_shared_m2"]

        y_hat_spec_m1_un = out_un["pred_spec_m1"]
        y_hat_spec_m2_un = out_un["pred_spec_m2"]


        shared_m1_ft_final_un = out_un["shared_m1_ft"]
        shared_m2_ft_final_un = out_un["shared_m2_ft"]
        spec_m1_ft_final_un = out_un["spec_m1_ft"]
        spec_m2_ft_final_un = out_un["spec_m2_ft"]

        loss_sup_fused = self.loss(y_hat_shared[0], y)
        loss_sup_shared_m1 = self.loss(y_hat_shared_m1[0], y)
        loss_sup_shared_m2  = self.loss(y_hat_shared_m2[0], y)

        loss_sup_spec_m1 = self.loss(y_hat_spec_m1[0], y)
        loss_sup_spec_m2  = self.loss(y_hat_spec_m2[0], y)

        loss_sup = loss_sup_fused + loss_sup_shared_m1 + loss_sup_shared_m2 + loss_sup_spec_m1 + loss_sup_spec_m2
        # loss_sup = self.deeploss(y_hat_shared, y)
        # print("loss_sup----:", loss_sup.item()) 
        # print("loss_sup_fused----:", loss_sup_fused.item()) 
        # print("loss_sup_shared_m1,m2----:", loss_sup_shared_m1.item(), loss_sup_shared_m2.item()) 
        # print("loss_sup_spec_m1,m2----:", loss_sup_spec_m1.item(), loss_sup_spec_m2.item()) 


        loss_orth_m1 = self.cosine_orthogonal_loss(shared_m1_ft_final, spec_m1_ft_final)
        loss_orth_m2 = self.cosine_orthogonal_loss(shared_m2_ft_final, spec_m2_ft_final)
        loss_shared_align = self.shared_alignment_loss(shared_m1_ft_final, shared_m2_ft_final)
        loss_dis = loss_orth_m1 + loss_orth_m2 + loss_shared_align
        # print("loss_orth----:", loss_orth_m1 + loss_orth_m2) 
        # print("loss_shared_align----:", loss_shared_align) 
       


        loss_orth_m1_un = self.cosine_orthogonal_loss(shared_m1_ft_final_un, spec_m1_ft_final_un)
        loss_orth_m2_un = self.cosine_orthogonal_loss(shared_m2_ft_final_un, spec_m2_ft_final_un)
        loss_shared_align_un = self.shared_alignment_loss(shared_m1_ft_final_un, shared_m2_ft_final_un)
        loss_dis_un = loss_orth_m1_un + loss_orth_m2_un + loss_shared_align_un
        # print("loss_orth_un----:", loss_orth_m1_un + loss_orth_m2_un) 
        # print("loss_shared_align_un----:", loss_shared_align_un) 
        
       
        consistency_weight = self.get_current_consistency_weight(self.current_epoch)
        # print("consistency_weight----:", consistency_weight)


        # loss_pred_consistency_spec = self.prediction_consistency_loss(y_hat_spec_m1_un[0], y_hat_spec_m2_un[0])
        loss_pred_consistency_shared = self.prediction_consistency_loss(y_hat_shared_m1_un[0], y_hat_shared_m2_un[0])
        loss_pred_cons = loss_pred_consistency_shared
        print("loss_pred_cons----:", loss_pred_cons.item())


        loss_noise_cons = self.noise_consistency(y_hat_shared_un[0],y_hat_shared_un_aug[0])
        print("loss_noise_cons----:", loss_noise_cons.item() )

        loss_fused2spec_cons= self.noise_consistency(y_hat_shared_un[0],y_hat_spec_m1_un[0])
        loss_fused2shared_cons= self.noise_consistency(y_hat_shared_un[0],y_hat_shared_m1_un[0])
        loss_fused2spec_cons2= self.noise_consistency(y_hat_shared_un[0],y_hat_spec_m2_un[0])
        loss_fused2shared_cons2= self.noise_consistency(y_hat_shared_un[0],y_hat_shared_m2_un[0])
        # loss_fused2shared_spec_cons = (loss_fused2spec_cons + loss_fused2shared_cons) / 2
        loss_fused2shared_spec_cons = (loss_fused2spec_cons + loss_fused2shared_cons + loss_fused2spec_cons2 + loss_fused2shared_cons2) / 4   
        print("loss_fused2shared_spec_cons----:", loss_fused2shared_spec_cons.item() )

        # loss = loss_sup_fused 
        # loss = loss_sup 
        # loss = loss_sup + 0.01*loss_dis + 0.01*loss_dis_un # baseline0.01
        # loss = loss_sup + 0.1*loss_dis + 0.1*loss_dis_un # baseline0.1
        # loss = loss_sup_fused + 0.1*loss_dis #supervised_withdisentangle
        loss = loss_sup + 0.01*loss_dis + 0.01*loss_dis_un + consistency_weight *(loss_noise_cons + loss_pred_cons  + loss_fused2shared_spec_cons)


        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["label"]
        y_hat = self.predict_step(batch, batch_idx)

         # ===== Dice/iou =====
        dice = self.dice_metric(y_hat, y)
        iou  = self.iou_metric(y_hat, y)

        # ===== HD95 & ASD =====
        hd95 = self.hd95_metric(y_hat, y)
        asd = self.asd_metric(y_hat, y)

        case_id = batch["id"][0]
        # print("validation case id: ", batch["id"])

        self.val_outputs.append({
            "id": case_id,
            "dice": dice.detach().cpu().item(),
            "iou":  iou.detach().cpu().item(),
            "hd95": hd95.detach().cpu().item(),
            "asd": asd.detach().cpu().item(),
        })

        self.val_preds.append({
        "id": case_id,
        "pred": y_hat.cpu(),
        "gt": y.cpu(),
        "affine": batch["image"].meta["affine"][0].cpu(),
        })
        return  dice

    def _gather_validation_items(self, items):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(gathered, items)
            all_items = []
            for g in gathered:
                all_items.extend(g)
            return all_items
        return items

    def save_val_per_sample_csv(self):
        all_val_outputs = self._gather_validation_items(self.val_outputs)

        if not self.trainer.is_global_zero or len(all_val_outputs) == 0:
            return

        per_sample_dir = os.path.join(self.logger.log_dir, "val_per_sample")
        os.makedirs(per_sample_dir, exist_ok=True)

        per_sample_csv = os.path.join(
            per_sample_dir, f"val_epoch_{self.current_epoch}.csv"
        )

        with open(per_sample_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_val_outputs[0].keys())
            writer.writeheader()
            writer.writerows(all_val_outputs)

    def save_val_predictions(self):
        all_preds = self._gather_validation_items(self.val_preds)

        if not self.trainer.is_global_zero:
            return

        save_dir = os.path.join(
            self.logger.log_dir,
            "predictions",
            # f"epoch_{self.current_epoch:03d}",
        )
        os.makedirs(save_dir, exist_ok=True)

        for item in all_preds:
            pred_np = item["pred"].squeeze().numpy().astype(np.uint8)
            gt_np = item["gt"].squeeze().numpy().astype(np.uint8)

            affine = item["affine"].numpy()

            save_path = os.path.join(save_dir, f"{item['id']}.nii.gz")
            nib.save(nib.Nifti1Image(pred_np, affine), save_path)
            # nib.save(nib.Nifti1Image(gt_np, affine), save_path)

    def save_val_mean_csv(self, mean_dice, mean_iou, mean_hd95, mean_asd, lr):
        mean_row = {
            "epoch": self.current_epoch,
            "lr": lr,
            "mean_dice": mean_dice.item(),
            "mean_iou":  mean_iou.item(),
            "mean_hd95": mean_hd95.item(),
            "mean_asd":  mean_asd.item(),
        }

        if not self.trainer.is_global_zero:
            return

        mean_csv = os.path.join(self.logger.log_dir, "val_metrics_mean.csv")
        file_exists = os.path.isfile(mean_csv)

        with open(mean_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=mean_row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(mean_row)

    def validation_epoch_end(self, val_outputs):
        mean_dice, _ = self.dice_metric.aggregate()
        mean_iou     = self.iou_metric.aggregate()  
        mean_hd95 = self.hd95_metric.aggregate()
        mean_asd = self.asd_metric.aggregate()

        self.log("val_dice", mean_dice, prog_bar=True, sync_dist=True)
        self.log("val_iou", mean_iou, prog_bar=True, sync_dist=True)
        self.log("val_hd95", mean_hd95, prog_bar=True, sync_dist=True)
        self.log("val_asd", mean_asd, prog_bar=True, sync_dist=True)

        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hd95_metric.reset()
        self.asd_metric.reset()

        if len(self.val_outputs) == 0:
            return
        # ===== learning rate =====
        if self.trainer.optimizers:
                lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        else:
                lr = None   #  0.0 / "N/A"

        self.save_val_per_sample_csv()
        self.save_val_predictions()
        self.save_val_mean_csv(mean_dice, mean_iou, mean_hd95, mean_asd, lr)

        self.val_outputs.clear()
        self.val_preds.clear()

        val_results = {
        "val_dice": mean_dice,
        "val_iou":  mean_iou,
        "val_hd95": mean_hd95,
        "val_asd":  mean_asd,
        }
       
        return val_results
