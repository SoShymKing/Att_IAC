import json
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from einops import rearrange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from monai.networks.nets import SwinUNETR, AttentionUnet
from monai.losses import DiceCELoss, MaskedDiceLoss
from monai.data import Dataset, DataLoader
from monai.transforms import (
    MapTransform, Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    RandCropByPosNegLabeld, ToTensord, EnsureTyped, RandFlipd, RandAffined
)
from loss_fn import FocalTverskyBoundaryLoss
import torch.nn.functional as F

def load_checkpoint_weights(model, checkpoint_path, strict=False, prefix_to_strip=None):
    print(f"Loading weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, weights_only=False)
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
        
    if prefix_to_strip:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_strip):
                new_key = k[len(prefix_to_strip):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        state_dict = new_state_dict
    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded with strict={strict}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model

def first_load_checkpoint_weights_SwinUNETR(model, checkpoint_path, strict=False, prefix_to_strip=None):
    def expand_conv_weight(weight):
        return weight.repeat(1, 2, 1, 1, 1) / 2  # Average over the new 2 channels

    print(f"Loading weights from {checkpoint_path}")
    state = torch.load(checkpoint_path, weights_only=False)
    if "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
        
    if prefix_to_strip:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_strip):
                new_key = k[len(prefix_to_strip):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        state_dict = new_state_dict        
    keys_to_expand = [
        "swinViT.patch_embed.proj.weight",
        "encoder1.layer.conv1.conv.weight",
        "encoder1.layer.conv3.conv.weight"
    ]
    for key in keys_to_expand:
        if key in state_dict:
            print(f"Expanding {key} from {state_dict[key].shape} to 2-channel input.")
            state_dict[key] = expand_conv_weight(state_dict[key])
    
    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded with strict={strict}")
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    return model


class PrintBestEpoch(Callback):
    def __init__(self, checkpoint_callback):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.last_ckpt = None
    def on_train_epoch_end(self, trainer, pl_module):
        current_ckpt = self.checkpoint_callback.best_model_path
        if current_ckpt and current_ckpt != self.last_ckpt:
            print(f"New best checkpoint saved: {current_ckpt}")
            self.last_ckpt = current_ckpt
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoint",           # Directory to save checkpoints
    filename="cmb-{epoch:02d}-{val_loss:.2f}-{train_epoch_loss:.2f}",     # Checkpoint filename format
    save_top_k=1,                    # Only save the best model
    monitor="val_loss",              # Metric to monitor
    mode="min",                      # Save the model with minimum val loss
    save_weights_only=False          # save weights
)
print_best_ckpt = PrintBestEpoch(checkpoint_callback)

class PrintNewestEpoch(Callback):
    def __init__(self, checkpoint_callback):
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.last_ckpt = None
    def cleanup_old_checkpoints(self, dir_path="checkpoint"):
        pattern = os.path.join(dir_path, "cmb-latest-epoch=*.ckpt")
        ckpts = glob.glob(pattern)
        for i in ckpts:
            if i != self.last_ckpt:
                os.remove(i)
    def on_train_epoch_end(self, trainer, pl_module):
        self.cleanup_old_checkpoints
        # current_ckpt = self.checkpoint_callback.best_model_path
        # if current_ckpt and current_ckpt != self.last_ckpt:
        #     print(f"New Newest checkpoint saved: {current_ckpt}")
        #     self.last_ckpt = current_ckpt
newest_checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoint",           # Directory to save checkpoints
    save_top_k=1,
    monitor=None,               # Not tracking metric
    every_n_epochs=1,           # Save each epoch
    save_last=False,             # Also save as 'last.ckpt'
    filename="cmb-latest-{epoch:02d}",          # Overwritten each epoch
    save_weights_only=False          # save weights
)
cleanup_ckpt = PrintNewestEpoch(newest_checkpoint_callback)


class LoadClickPoints(MapTransform):
    def __init__(self, keys="click", output_key="coord", nums=10):
        super().__init__(keys)
        self.output_key = output_key
        self.nums = nums

    def __call__(self, data):
        d = dict(data)
        path = d[self.keys[0]]  # "coord" → path to click_data.json
        with open(path, "r") as f:
            click_data = json.load(f)

        points = [p["point"] for p in click_data["points"][:self.nums]]
        d[self.output_key] = points  # Replace file path with list of coords
        return d
        
def create_guide_mask(shape, points, std=1.0):
    mask = np.zeros(shape, dtype=np.uint8)
    for z, y, x in points:
        # Add Gaussian noise to each coordinate
        z_n = int(np.clip(z + np.random.normal(0, std), 0, shape[0] - 1))
        y_n = int(np.clip(y + np.random.normal(0, std), 0, shape[1] - 1))
        x_n = int(np.clip(x + np.random.normal(0, std), 0, shape[2] - 1))
        mask[z_n, y_n, x_n] = 1
    return mask

class AddNoisyGuideMask(MapTransform):
    def __init__(self, image_key="image", coord_key="coord", output_key="guide", std=1.0):
        super().__init__(keys=image_key)
        self.coord_key = coord_key
        self.output_key = output_key
        self.std = std
    def __call__(self, data):
        d = dict(data)
        shape = d[self.keys[0]].shape[1:]  # (C, D, H, W) → D, H, W
        coords = d[self.coord_key]
        d[self.output_key] = create_guide_mask(shape, coords, self.std)[None]  # (1, D, H, W)
        return d
        
class BinaryLabelFromMaskd(MapTransform):
    """
    Converts label to binary: 1 if value in [3, 4], else 0
    """
    def __call__(self, data):
        d = dict(data)
        label = d[self.keys[0]]  # Assumes shape: (H, W, D) or (1, H, W, D)
        if label.ndim == 4:  # If (1, H, W, D), remove channel
            label = label[0]
        binary = np.isin(label, [3, 4]).astype(np.uint8)
        d[self.keys[0]] = binary[None]  # Add channel back → (1, H, W, D)
        return d
        
class GlobalContextFromResizedd(MapTransform):
    """
    Create a global context channel by resizing the full image to (96,96,96).
    Stores it at out_key (e.g., 'gctx'). Runs on CPU in the transform pipeline.
    """
    def __init__(self, image_key="image", out_key="gctx", size=(96,96,96)):
        super().__init__(keys=image_key)
        self.out_key = out_key
        self.size = size

    @torch.no_grad()
    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]  # [C,D,H,W], numpy or torch
        t = torch.from_numpy(img) if isinstance(img, np.ndarray) else img
        t = t.float().unsqueeze(0)          # [1,C,D,H,W]
        gctx = F.interpolate(t, size=self.size, mode="trilinear", align_corners=False)[0]  # [C,96,96,96]
        # (optional) compress C>1 to 1 channel if needed
        d[self.out_key] = gctx if not isinstance(img, np.ndarray) else gctx.numpy()
        return d

data_transforms = Compose([
    LoadImaged(keys=["image","label"]),
    EnsureChannelFirstd(keys=["image","label"]),
    LoadClickPoints(keys="click", output_key="coord", nums=10),
    AddNoisyGuideMask(image_key="image", coord_key="coord", output_key="guide", std=1.5),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0),
    # CopyItemsd(keys=["image"], times=1, names=["whole"]),
    # ResizeWithPadOrCropd(keys=["whole"], spatial_size=(128,128,128)),
    BinaryLabelFromMaskd(keys=["label"]),
    # RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
    # RandFlipd(keys=["label"], spatial_axis=0, prob=0.5),
    # RandFlipd(keys=["guide"], spatial_axis=0, prob=0.5),
    GlobalContextFromResizedd(image_key="image", out_key="gctx", size=(96,96,96)),
    RandFlipd(keys=["image","label","guide", "gctx"], spatial_axis=0, prob=0.5),
    # RandAffined(
    #     keys=["image", "label", "guide"],
    #     prob=0.5,
    #     rotate_range=(  # rotation in radians per axis
    #         np.pi / 2,  # x-axis (<≈ 90°)
    #         np.pi / 2,  # y-axis (<≈ 90°)
    #         np.pi / 2   # z-axis (<≈ 90°)
    #     ),
    #     padding_mode="border",      # or "zeros", "reflection"
    #     mode=("bilinear", "nearest", "nearest")  # interpolation per key
    # ),
    RandCropByPosNegLabeld(
        keys=["image", "guide", "label"],
        label_key="guide",
        # spatial_size=(96, 96, 96),
        spatial_size=(96, 96, 96),
        pos=3, neg=1,
        num_samples=4
    ),
    ToTensord(keys=["image", "guide", "label", "gctx"]),
    EnsureTyped(keys=["image", "guide", "label", "gctx"], dtype=torch.float32)
])

with open("./data/train_data.json") as f:
    train_dicts = json.load(f)
with open("./data/val_data.json") as f:
    val_dicts = json.load(f)
train_ds = Dataset(data=train_dicts, transform=data_transforms)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)#16
val_ds = Dataset(data=val_dicts, transform=data_transforms)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)#16


class CombinedModel(pl.LightningModule):
    def __init__(self, model_a, model_b, loss_fn):
        super().__init__()
        self.model_a = model_a  # Pretrained model A
        self.model_b = model_b  # Pretrained model B
        self.loss_fn = loss_fn

        for param in self.model_b.parameters():
           param.requires_grad = True 

        for param in self.model_a.parameters():
           param.requires_grad = True
    def forward(self, x, y, z):
        a_out = self.model_a(y)
        foreground_prob = torch.softmax(a_out, dim=1)[:, 1:2]  # [B, 1, D, H, W]
        combined_input = torch.cat([x, foreground_prob, z], dim=1) #torch.cat([x, foreground_prob, z], dim=1)
        b_out = self.model_b(combined_input)  # or use torch.cat([x, a_out], dim=1)

        # a_out = self.model_a(y)
        # # foreground_prob = torch.softmax(a_out, dim=1)[:, 1:2]  # [B, 1, D, H, W]
        # # combined_input = torch.cat([y, foreground_prob], dim=1)
        # b_out = self.model_b(torch.cat([x, a_out], dim=1))  # or use torch.cat([x, a_out], dim=1)
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
        image = batch["image"].to(self.device, non_blocking=True)  # shape: [B, 1, D, H, W]
        guide = batch["guide"].to(self.device, non_blocking=True)  # shape: [B, 1, D, H, W]
        label = batch["label"].to(self.device, non_blocking=True)
        gctx = batch["gctx"].to(self.device, non_blocking=True)
        pred = self.forward(image, guide, gctx)
        loss = self.loss_fn(pred, label)
        ####
        self.log("val_loss", loss, prog_bar=True, batch_size=1)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-5)        
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_outputs).mean()
        self.log("train_epoch_loss", avg_loss, prog_bar=True, batch_size=1)
        self.training_step_outputs.clear()
        
    def forward_full_then_patch(self, full_img):
        # 1) global context
        x_low = F.interpolate(full_img, scale_factor=1/self.downscale, mode="trilinear", align_corners=False)
        g_low = self.global_enc(x_low)  # [B, Cg, d, h, w]
        g_full = F.interpolate(g_low, size=full_img.shape[2:], mode="trilinear", align_corners=False)

        # 2) concat and feed local model (can be used with sliding window)
        x_cat = torch.cat([full_img, g_full], dim=1)  # in_ch must be 1 + Cg
        return self.local(x_cat), x_cat  # (logits, concatenated input to use with SWI)


device = "cuda" if torch.cuda.is_available() else "cpu"
combined_model = CombinedModel(
    model_a=AttentionUnet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2)  # must have len = len(channels) - 1
        ), 
    model_b=SwinUNETR(
            in_channels=3,
            out_channels=2,
            feature_size=48,
            spatial_dims=3,
        ),
    loss_fn = DiceCELoss(
        # sigmoid=True,
        lambda_dice=1.0,
        lambda_ce=1.0, 
        include_background=False,
        to_onehot_y=True,
        softmax=True
    )
    # loss_fn = FocalTverskyBoundaryLoss(
    # mode="multiclass",     # binary
    # from_logits=True,
    # alpha=0.3, beta=0.7,   # FN(미검출) 더 강하게 벌주기
    # gamma=4/3,             # 어려운 픽셀 강조
    # lambda_ft=1.0, lambda_bd=1.0,
    # include_background=False,
    # )
)
ckpt = torch.load("./checkpoint/cmb-epoch=93-val_loss=0.10-train_epoch_loss=0.10.ckpt", weights_only=False)

combined_model.load_state_dict(ckpt["state_dict"])

torch.set_float32_matmul_precision('high') #'medium' | 

trainer = Trainer(
    devices=1,
    max_epochs=10000, 
    accelerator=device, 
    callbacks=[checkpoint_callback, print_best_ckpt, newest_checkpoint_callback, cleanup_ckpt]
)

trainer.fit(combined_model, train_loader, val_loader)