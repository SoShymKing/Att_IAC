import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from monai.networks.nets import SwinUNETR, AttentionUnet
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

class CombinedModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model_a = AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2) 
        )  # Pretrained model A
        self.model_b = SwinUNETR(
            in_channels=3,
            out_channels=2,
            feature_size=48,
            spatial_dims=3,
        )  # Pretrained model B
        self.loss_fn = DiceCELoss(
            lambda_dice=1.0,
            lambda_ce=1.0, 
            include_background=False,
            to_onehot_y=True,
            softmax=True
        )

    def forward(self, x, y):
        a_out = self.model_a(x)
        foreground_prob = torch.softmax(a_out, dim=1)[:, 1:2]  # [B, 1, D, H, W]
        combined_input = torch.cat([y, foreground_prob], dim=1)
        b_out = self.model_b(combined_input)  # or use torch.cat([x, a_out], dim=1)
        return b_out

    def training_step(self, batch, batch_idx):
        image = batch["image"].to(self.device, non_blocking=True)  # shape: [B, 1, D, H, W]
        guide = batch["guide"].to(self.device, non_blocking=True)  # shape: [B, 1, D, H, W]
        label = batch["label"].to(self.device, non_blocking=True)
        gctx = batch["gctx"].to(self.device, non_blocking=True)
        pred = self.forward(image, guide, gctx)
        loss = self.loss_fn(pred, label)
        
        self.log("train_loss", loss, batch_size=1)
        if not hasattr(self, "training_step_outputs"):
            self.training_step_outputs = []
        self.training_step_outputs.append(loss.detach())
        return loss
    def validation_step(self, batch, batch_idx):
        img = batch["image"].to(self.device, non_blocking=True)  # shape: [B, 1, D, H, W]
        gid = batch["guide"].to(self.device, non_blocking=True)  # shape: [B, 1, D, H, W]
        label = batch["label"].to(self.device, non_blocking=True)
        # print(f"img shape{img.shape}, gid shape{gid.shape}")
        logits = self.inf_full_vol(img, gid, roi=(96,96,96), overlap=0.5)
        loss = self.loss_fn(logits, label)
        ####
        self.log("val_loss", loss, prog_bar=True, batch_size=1)
        return loss
    def logits_to_prob(self, logits):
        prob = torch.softmax(logits, dim=1)[:, 1:2]         # foreground prob [1,1,D,H,W]
        mask = (prob > 0.5).to(torch.uint8) 
        return mask
    def inf_full_vol(self, full_img, full_guide, roi=(96,96,96), overlap=0.5):
        self.model_a.eval()
        self.model_b.eval()

        gctx_roi = F.interpolate(full_img, size=roi, mode="trilinear", align_corners=False)
        c1 = full_img.shape[1]
        def pred_fn(x_patch):
            im = x_patch[:, :c1]
            gid = x_patch[:, c1:]
            a_out = self.model_a(gid)
            fg = torch.softmax(a_out, dim=1)[:, 1:2]  # [B, 1, D, H, W]
            B = x_patch.shape[0]
            if B == 1:
                g = gctx_roi
            else:
                g = gctx_roi.repeat(B, 1, 1, 1, 1)  # 실제 메모리 생성
            g = g.to(x_patch.device, x_patch.dtype)
            x_cat = torch.cat([im, fg, g], dim=1).contiguous()
            b_out = self.model_b(x_cat)
            return b_out 
        swi_input = torch.cat([full_img, full_guide], dim=1)  # [1,2,D,H,W]  
        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=swi_input,
                roi_size=roi,
                sw_batch_size=1,
                predictor=pred_fn,
                overlap=overlap,
                mode="gaussian",   # optional blending
            ) 
        return logits
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return  [optimizer], [scheduler]
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("train_epoch_loss", avg_loss, prog_bar=True, batch_size=1)
        self.training_step_outputs.clear()