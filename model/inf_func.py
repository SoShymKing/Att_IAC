import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from model.model import CombinedModel
from typing import Dict

def create_guide_mask(shape, points, std=1.0):
    mask = np.zeros(shape, dtype=np.uint8)
    for z, y, x in points:
        # Add Gaussian noise to each coordinate
        z_n = int(np.clip(z, 0, shape[0] - 1))
        y_n = int(np.clip(y, 0, shape[1] - 1))
        x_n = int(np.clip(x, 0, shape[2] - 1))
        mask[z_n, y_n, x_n] = 1
    return mask

def load_click_points(shape, click_data):
    points = [p["point"] for p in click_data["points"]]
    mask = create_guide_mask(shape, points)
    return mask, points

def load_click_points_diff(shape, click_data):
    left = [p["point"] for p in click_data["points"] if p["name"] == "Left_IAC"]
    right = [p["point"] for p in click_data["points"] if p["name"] == "Right_IAC"]
    left_mask = create_guide_mask(shape, left)
    right_mask = create_guide_mask(shape, right)
    return left_mask, right_mask, left, right

def scale_intensity(img, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0):
    img_d = torch.as_tensor(img)
    img_d = img_d.float()
    img_d = (img_d - a_min) / (a_max - a_min)
    img_d = img_d.clamp_(0, 1)           # clip=True behavior
    return img_d * (b_max - b_min) + b_min

@torch.no_grad()
def infer_full_volume(model, full_img, full_guide, roi=(96,96,96), overlap=0.5, a_outputs_logits=True):
    """
    model_local: expects 3 channels (image, guide, gctx) -> logits [B,C,d,h,w]
    full_img   : [1,1,D,H,W] float
    full_guide : [1,1,D,H,W] float (or 0/1)
    """

    device = next(model.parameters()).device
    full_img   = full_img.to(device).float()
    full_guide = full_guide.to(device).float()

    # Build global context ONCE from the full image: resize to ROI size (96³ here)
    gctx_roi = F.interpolate(full_img, size=roi, mode="trilinear", align_corners=False)  # [1,1,96,96,96]

    c1 = full_img.shape[1]  # channels of image
    # sliding-window will tile over [image, guide]; we append gctx per patch
    def pred_fn(x_patch):
        im = x_patch[:, :c1]
        gid = x_patch[:, c1:]
        a_out = model.model_a(gid)
        fg = torch.softmax(a_out, dim=1)[:, 1:2]  # [B, 1, D, H, W]
        
        B = x_patch.shape[0]
        g = gctx_roi.expand(B, -1, -1, -1, -1).to(x_patch.device, x_patch.dtype)  # [B,1,96,96,96]
        
        x_cat = torch.cat([im, fg, g], dim=1)
        b_out = model.model_b(x_cat)
        return b_out     #torch.sigmoid(b_out)

    # Prepare 2-channel input for SWI (image+guide)
    swi_input = torch.cat([full_img, full_guide], dim=1)  # [1,2,D,H,W]

    model.model_a.eval()
    model.model_b.eval()
    
    logits = sliding_window_inference(
        inputs=swi_input,
        roi_size=roi,
        sw_batch_size=1,
        predictor=pred_fn,
        overlap=overlap,
        mode="gaussian",   # optional blending
    )  # [1,C,D,H,W]

    # Postproc: choose your flavor
    # For 2-class logits:
    prob = torch.softmax(logits, dim=1)[:, 1:2]         # foreground prob [1,1,D,H,W]
    mask = (prob > 0.5).to(torch.uint8)                 # binary mask   [1,1,D,H,W]
    return logits, prob, mask

def m_infer_full_volume(model, full_img, full_guide, roi=(96,96,96), overlap=0.5):
    logits  = model.inf_full_vol(full_img, full_guide, roi=roi, overlap=overlap)
    # For 2-class logits:
    prob = torch.softmax(logits, dim=1)[:, 1:2]         # foreground prob [1,1,D,H,W]
    mask = (prob > 0.5).to(torch.uint8)                 # binary mask   [1,1,D,H,W]
    return logits, prob, mask

def inference_iac(input_tensor: torch.Tensor, clicks_data: Dict, device = "cpu", ckpt = "ev_checkpoint_val_loss=0.10.ckpt") -> np.ndarray:
    guide, points = load_click_points(input_tensor[0].shape, clicks_data)
    # print(points)
    image_data = scale_intensity(input_tensor, -1000, 1000, 0.0, 1.0)
    full_img = image_data.to(torch.float32).unsqueeze(0) # shape: [1, 1, H, W, D]
    full_guide = torch.tensor(guide, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W, D]
    model = CombinedModel()
    model.to(device)
    full_img = full_img.to(device)
    full_guide = full_guide.to(device)
    ckpt = torch.load(ckpt, map_location=device, weights_only=False)
    incompat = model.load_state_dict(ckpt["state_dict"])

    logits, prob, mask = m_infer_full_volume(model, full_img, full_guide, roi=(96,96,96), overlap=0.5)
    print(f"mask none zero: {torch.sum(mask != 0).item()}")
    print(f"logits none zero: {torch.sum(logits != 0).item()}")
    print(f"prob none zero: {torch.sum(prob != 0).item()}")
    print(f"prob max: {prob.max()}, prob min: {prob.min()}")
    # logits, prob, mask = infer_full_volume(model, full_img, full_guide, roi=(96,96,96), overlap=0.5)
    # mask_np = mask.cpu().numpy()
    mask_np = mask.squeeze()
    return mask_np
    