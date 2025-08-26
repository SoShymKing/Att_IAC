from model.inf_func import inference_iac
import numpy as np
import torch
from typing import Dict, Tuple, List

def create_side_mask(shape, points):
    mask = np.zeros(shape, dtype=np.uint8)
    mean_x = 0
    mean_y = 0
    mean_z = 0
    for z, y, x in points:
        mean_z += int(z)
        mean_y += int(y)
        mean_x += int(x)
    if len(points) > 0:
        mean_z //= len(points)
        mean_y //= len(points)
        mean_x //= len(points)
        if mean_x < (shape[0] - 1)//2:
            mask[:(shape[0] - 1)//2,:,:] = 1
        else:
            mask[(shape[0] - 1)//2:,:,:] = 1
    return mask
def create_def_mask(shape):
    print(f"shape: {shape}")
    basic_l = np.zeros(shape, dtype=np.uint8)
    basic_r = np.zeros(shape, dtype=np.uint8)
    basic_l[:(shape[0] - 1)//2, :,:] = 1
    basic_r[(shape[0] - 1)//2:, :,:] = 1
    return basic_l, basic_r

def load_click_points_diff(shape, clicks_data):
    # points = [p for p in clicks_data["points"][:10]]
    # left = [p["point"] for p in points if p["name"] == "Left_IAC"]
    # right = [p["point"] for p in points if p["name"] == "Right_IAC"]
    left = [p["point"] for p in clicks_data["points"] if p["name"] == "Left_IAC"]
    right = [p["point"] for p in clicks_data["points"] if p["name"] == "Right_IAC"]
    basic_l, basic_r = create_def_mask(shape)
    if len(left) == 0 :
        left_mask = basic_l
    else:
        left_mask = create_side_mask(shape, left)
    if len(right) == 0:
        right_mask = basic_r
    else:
        right_mask = create_side_mask(shape, right)
    return left_mask, right_mask, left, right


def simple_mask(shape):
    left_mask = np.zeros(shape, dtype=np.uint8)
    right_mask = np.zeros(shape, dtype=np.uint8)
    return left_mask, right_mask

def left_right_split_tensor(input_tensor: torch.Tensor, clicks_data: Dict) -> List[torch.Tensor]:
    left_mask, right_mask, left_points, right_points = load_click_points_diff(input_tensor.shape, clicks_data)
    left_mask = torch.from_numpy(left_mask).to(input_tensor.device)
    right_mask = torch.from_numpy(right_mask).to(input_tensor.device)
    left_tensor = input_tensor * left_mask
    right_tensor = input_tensor * right_mask
    return [left_tensor, right_tensor]

LABELS = {
    # "Left Inferior Alveolar Canal": 1,
    # "Right Inferior Alveolar Canal": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    # "Right Inferior Alveolar Canal": 4,
}

def combine_left_right_tensors(left_tensor: torch.Tensor, right_tensor: torch.Tensor) -> torch.Tensor:
    left_tensor = left_tensor * LABELS["Left Inferior Alveolar Canal"]
    right_tensor = right_tensor * LABELS["Right Inferior Alveolar Canal"]
    return left_tensor + right_tensor

def get_bounding_boxes(mask: torch.Tensor, values=(3, 4)):
    bboxes = {}
    for v in values:
        coords = (mask == v).nonzero(as_tuple=False)
        if coords.numel() == 0:
            bboxes[v] = None
            continue

        z_min, y_min, x_min = coords.min(dim=0).values.tolist()
        z_max, y_max, x_max = coords.max(dim=0).values.tolist()
        bboxes[v] = ( x_min, x_max, y_min, y_max, z_min, z_max)

    return bboxes

def IAC_SegmentationAlgorithm(input_tensor: torch.Tensor, clicks_data: Dict, device: torch.device) -> np.ndarray:
    # mask = inference_iac(input_tensor*2100, clicks_data, ckpt = "/opt/app/model/ev_checkpoint_val_loss=0.10.ckpt", device=device)
    # mask = inference_iac(input_tensor.permute(0, 3, 2, 1), clicks_data, ckpt = "/opt/app/model/ev_checkpoint_val_loss=0.10.ckpt", device=device)
    mask = inference_iac(input_tensor.permute(0, 3, 2, 1)*2100, clicks_data, ckpt = "/opt/app/model/ev_checkpoint_val_loss=0.10.ckpt", device=device)
    # mask = inference_iac(input_tensor, clicks_data, ckpt = "/opt/app/model/ev_checkpoint_val_loss=0.10.ckpt", device=device)
    l, r = left_right_split_tensor(mask, clicks_data)
    combined_mask = combine_left_right_tensors(l, r).permute(2, 1, 0)
    # combined_mask = combine_left_right_tensors(l, r)
    print(f"combined_mask shape: {combined_mask.shape}")
    print(f"combined_mask ones: {torch.sum(combined_mask == 3).item()}")
    print(f"combined_mask twos: {torch.sum(combined_mask == 4).item()}")
    print("Bounding boxes:", get_bounding_boxes(combined_mask, values=(3, 4)))
    return combined_mask.cpu().numpy().astype(np.uint8)