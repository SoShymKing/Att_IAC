import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, HausdorffDTLoss

# ---- (optional) SDM 생성: scipy 사용 ----
# 설치: pip install scipy
def compute_sdf_from_masks(mask_one_hot: torch.Tensor) -> torch.Tensor:
    """
    mask_one_hot: (B, C, [D,] H, W), 0/1 one-hot
    return: signed distance map with same shape (float32)
    """
    try:
        import numpy as np
        from scipy.ndimage import distance_transform_edt as edt
    except Exception as e:
        raise ImportError(
            "SciPy가 필요합니다. `pip install scipy` 후 다시 시도하거나, "
            "`forward(..., sdf_target=...)`로 미리 계산한 SDF를 넣어주세요."
        ) from e

    device = mask_one_hot.device
    ndim = mask_one_hot.dim() - 2
    ms = mask_one_hot.detach().cpu().numpy().astype(np.uint8)

    sdf = np.zeros_like(ms, dtype=np.float32)
    # 채널별 SDM
    it = np.nditer(np.zeros(ms.shape[:2], dtype=np.uint8), flags=['multi_index'])
    while not it.finished:
        b, c = it.multi_index
        m = ms[b, c]  # [D,]H,W
        # 내부/외부 거리
        posmask = m.astype(bool)
        negmask = ~posmask
        if ndim == 3:
            dist_out = edt(negmask)
            dist_in  = edt(posmask)
        else:  # 2D
            dist_out = edt(negmask)
            dist_in  = edt(posmask)
        sdf[b, c] = dist_out - dist_in  # 바깥 +, 안쪽 -
        it.iternext()

    return torch.from_numpy(sdf).to(device)
    
class FocalTverskyLoss(nn.Module):
      def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6, include_background=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_background = include_background

      def forward(self, inputs, targets):
        """
        inputs: logits [B, C, ...]
        targets: class index [B, 1, ...] or one-hot [B, C, ...] (멀티클래스),
                 바이너리일 땐 [B, 1, ...] (0/1)
        """
        B, C = inputs.shape[:2]
        # 확률화
        if C == 1:  # binary
            probs = torch.sigmoid(inputs)
        else:
            probs = F.softmax(inputs, dim=1)
        # ---- 타깃 정규화 ----
        if C == 1:
            # binary: targets를 (B,1,...)로 맞추기
            if targets.ndim == inputs.ndim - 1:  # [B, ...,] -> 채널 추가
                targets = targets.unsqueeze(1)
            elif targets.ndim == inputs.ndim and targets.shape[1] != 1:
                # 만약 실수로 one-hot이 들어왔다면 전경 채널만 사용
                targets = targets[:, :1]
            targets = targets.float()
        else:
            # multiclass: (B,1,...)면 one-hot로 변환
            if targets.ndim == inputs.ndim and targets.shape[1] == 1:
                oh = F.one_hot(targets.squeeze(1).long(), num_classes=C)
                # [B, ..., C] -> [B, C, ...]
                perm = (0, -1) + tuple(range(1, oh.ndim - 1))
                targets = oh.permute(perm).contiguous().float()
            elif targets.ndim == inputs.ndim - 1:
                # [B, ...,] (채널없음) -> one-hot
                oh = F.one_hot(targets.long(), num_classes=C)
                perm = (0, -1) + tuple(range(1, oh.ndim - 1))
                targets = oh.permute(perm).contiguous().float()
            # 이미 one-hot이면 그대로 둔다

            # 배경 제외 옵션
            if not self.include_background:
                probs = probs[:, 1:]
                targets = targets[:, 1:]

        # ---- Tversky 계산 ----
        # (B, C or 1, ...) -> 펼침
        dims = tuple(range(2, probs.dim()))
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1 - targets)).sum(dim=dims)
        fn = ((1 - probs) * targets).sum(dim=dims)

        # 멀티클래스면 채널 평균
        if tp.ndim == 2:
            tp = tp.mean(dim=1)
            fp = fp.mean(dim=1)
            fn = fn.mean(dim=1)

        TI = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - TI) ** self.gamma
        return loss.mean()

        
class FocalTverskyBoundaryLoss(nn.Module):
    """
    Hybrid loss: Focal Tversky + Boundary (SDM 기반)
    loss = lambda_ft * FocalTversky + lambda_bd * Boundary

    Args:
        mode: 'multiclass' | 'multilabel' | 'binary'
        from_logits: True면 sigmoid/softmax 적용
        alpha, beta: Tversky 계수 (FN/FP 가중). 작은 물체 놓침(FN) 줄이려면 beta↑ (ex: alpha=0.3, beta=0.7)
        gamma: Focal 지수 (ex: 1.33 ~ 1.5)
        lambda_ft, lambda_bd: 두 손실 가중치
        include_background: 배경 채널(0)을 Dice/Tversky/Boundary에서 제외할지
        class_weights: (C,) 텐서. 채널별 가중
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(
        self,
        mode: str = "multiclass",
        from_logits: bool = True,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4/3,
        lambda_ft: float = 1.0,
        lambda_bd: float = 1.0,
        include_background: bool = False,
        class_weights: torch.Tensor | None = None,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        assert mode in {"multiclass", "multilabel", "binary"}
        assert reduction in {"mean", "sum", "none"}
        self.mode = mode
        self.from_logits = from_logits
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ft = lambda_ft
        self.lambda_bd = lambda_bd
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    # ---- helpers ----
    def _activate(self, pred):
        if not self.from_logits:
            return pred
        if self.mode in {"binary", "multilabel"}:
            return torch.sigmoid(pred)
        return F.softmax(pred, dim=1)

    def _to_one_hot(self, target, num_classes, like_pred):
        # target: (B,1,...) class index or already (B,C,...)
        if self.mode == "multiclass":
            if target.shape[1] == 1:
                oh = F.one_hot(target.squeeze(1).long(), num_classes=num_classes)
                # move last axis to channel
                perm = (0, -1) + tuple(range(1, target.dim()-1))
                oh = oh.permute(perm).contiguous().float()
            else:
                oh = target.float()
        else:  # binary/multilabel expects same shape as pred
            oh = target.float()
        # ensure shape exactly (B,C,...)
        if oh.shape[1] != num_classes:
            raise ValueError(f"one-hot shape mismatch: got C={oh.shape[1]} expected {num_classes}")
        # cast to same device/dtype
        return oh.to(like_pred)

    def _maybe_exclude_bg(self, prob, tgt, sdf=None):
        if self.include_background:
            return prob, tgt, sdf
        prob = prob[:, 1:]
        tgt  = tgt[:, 1:]
        if sdf is not None:
            sdf = sdf[:, 1:]
        return prob, tgt, sdf

    # ---- focal tversky ----
    def _focal_tversky(self, prob, tgt):
        # prob,tgt: (B,C,...) probs and one-hot targets
        dims = tuple(range(2, prob.dim()))
        tp = (prob * tgt).sum(dim=dims)
        fp = (prob * (1 - tgt)).sum(dim=dims)
        fn = ((1 - prob) * tgt).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)  # (B,C)
        ft = (1.0 - tversky) ** self.gamma  # (B,C)

        if self.class_weights is not None:
            w = self.class_weights.to(ft.dtype)
            if w.numel() == ft.shape[1]:
                ft = ft * w
            elif (not self.include_background) and w.numel() == ft.shape[1] + 1:
                ft = ft * w[1:]
            else:
                raise ValueError("class_weights size mismatch.")
        ft = ft.mean(dim=1)  # (B,)
        return ft

    # ---- boundary (SDM) loss ----
    def _boundary_loss(self, prob, sdf):
        """
        Kervadec et al. Boundary loss:
        L_bd = mean over batch and channels of <prob, sdf>_Omega / |Omega|
        실무에선 클래스별 평균(|phi| 정규화) 또는 공간 평균으로 안정화.
        여기서는 채널/공간 평균.
        """
        # prob,sdf: (B,C,...) ; sdf는 바깥 +, 안쪽 -
        # 바깥쪽(양수) 확률을 줄이고, 안쪽(음수)에서 확률을 키우도록 유도
        # 규모 안정화를 위해 절댓값으로 정규화 옵션도 가능하나 기본형 유지
        loss = (prob * sdf).mean(dim=tuple(range(1, prob.dim())))  # (B,)
        # 양수 쪽(외부) 가중이 커지면 불안정할 수 있어 작은 스케일로 clip 권장
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, sdf_target: torch.Tensor | None = None):
        """
        pred: (B,C,[D,]H,W) logits or probs
        target:
          - multiclass: (B,1,[D,]H,W) or (B,C,...) one-hot
          - binary/multilabel: (B,C,...) 0/1
        sdf_target (optional): (B,C,[D,]H,W) SDM. 없으면 내부에서 one-hot로부터 계산(Scipy 필요).
        """
        probs = self._activate(pred)
        B, C = probs.shape[:2]
        tgt_oh = self._to_one_hot(target, num_classes=C, like_pred=probs)

        # SDM 준비
        if sdf_target is None:
            sdf = compute_sdf_from_masks(tgt_oh)  # requires scipy
        else:
            sdf = sdf_target.to(probs)

        # 배경 제외 옵션
        probs, tgt_oh, sdf = self._maybe_exclude_bg(probs, tgt_oh, sdf)

        # --- Focal Tversky ---
        ft = self._focal_tversky(probs, tgt_oh)  # (B,)
        # --- Boundary ---
        bd = self._boundary_loss(probs, sdf)     # (B,)

        loss = self.lambda_ft * ft + self.lambda_bd * bd
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # (B,)




# ---------------------------
# clDice: topology / centerline 보조항 (간단 구현)
# ---------------------------
def soft_skeletonize(x: torch.Tensor, iters: int = 10) -> torch.Tensor:
    """
    x: (B,1,[D,]H,W) in [0,1]
    간단한 erosion-like 근사로 중심선 강조(매우 간이 버전).
    iters를 5~15 사이로 조절.
    """
    # 2D/3D 커널 자동 선택
    dims = x.dim() - 2  # 2 or 3
    if dims == 2:
        pool = F.max_pool2d
        k = 3
    elif dims == 3:
        pool = F.max_pool3d
        k = 3
    else:
        raise ValueError("soft_skeletonize expects 2D/3D tensors.")

    x = x.clamp(0, 1)
    one_minus = 1 - x
    for _ in range(iters):
        # 간단한 형태학적 thinning 근사
        eroded = pool(one_minus, kernel_size=k, stride=1, padding=k // 2)
        x = torch.clamp(x - torch.relu(eroded - one_minus), 0, 1)
    return x

def cldice_loss(prob_pred: torch.Tensor, prob_gt: torch.Tensor, eps: float = 1e-6, skel_iters: int = 10) -> torch.Tensor:
    """
    prob_pred, prob_gt: (B,1,[D,]H,W) in [0,1]
    반환: 스칼라 loss
    """
    Sk_p = soft_skeletonize(prob_pred, iters=skel_iters)
    Sk_g = soft_skeletonize(prob_gt,   iters=skel_iters)

    tprec = (Sk_p * prob_gt).sum(dim=tuple(range(2, prob_pred.dim()))) / (Sk_p.sum(dim=tuple(range(2, prob_pred.dim()))) + eps)
    tsens = (Sk_g * prob_pred).sum(dim=tuple(range(2, prob_pred.dim()))) / (Sk_g.sum(dim=tuple(range(2, prob_pred.dim()))) + eps)

    cl = 1.0 - (2 * tprec * tsens / (tprec + tsens + eps))
    return cl.mean()


# ---------------------------
# Combined Loss Module
# ---------------------------
class ConnectivityAwareSegLoss(nn.Module):
    """
    끊김(단절) 완화를 위한 조합 손실:
    L = DiceCE + λ1*FocalTversky + λ2*HausdorffDT + λ3*clDice

    Args:
        lambda_ftv: FocalTversky 가중치 (권장 0.1~0.3)
        lambda_hddt: HausdorffDT 가중치 (권장 0.1~0.3)
        lambda_cl: clDice 보조항 가중치 (권장 0.05~0.2)
        include_background: False 권장(전경만)
        ftv_alpha, ftv_beta, ftv_gamma: FocalTversky 파라미터 (FN 페널티↑: alpha<beta, gamma≈1.3)
        use_boundary_loss: HausdorffDT 대신 BoundaryLoss(SDM) 사용 시 True (라벨 SDM 필요)
        cl_skel_iters: clDice skeleton 반복 횟수
    """
    def __init__(
        self,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0, 
        to_onehot_y: bool = True,
        softmax: bool = True,
        lambda_ftv: float = 0.2,
        lambda_hddt: float = 0.2,
        lambda_cl: float = 0.1,
        include_background: bool = False,
        ftv_alpha: float = 0.3,
        ftv_beta: float = 0.7,
        ftv_gamma: float = 1.3,
        use_boundary_loss: bool = False,
        cl_skel_iters: int = 10,
    ):
        super().__init__()
        self.lambda_ftv = lambda_ftv
        self.lambda_hddt = lambda_hddt
        self.lambda_cl = lambda_cl
        self.include_background = include_background
        self.cl_skel_iters = cl_skel_iters
        self.use_boundary_loss = use_boundary_loss

        # 메인 손실
        # (출력 채널이 1이면 내부적으로 sigmoid 사용, 2 이상이면 softmax 경로 사용)
        self.dicece = DiceCELoss(
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce, 
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            include_background=include_background)

        # FN 완화
        self.ftv = FocalTverskyLoss(alpha=ftv_alpha, beta=ftv_beta, gamma=ftv_gamma, include_background=include_background)

        # 경계/거리
        if not use_boundary_loss:
            self.hddt = HausdorffDTLoss(
                include_background=include_background,  # 전경만
                to_onehot_y=to_onehot_y,         # 라벨 one-hot 변환 금지 (중요)
                # sigmoid=True,              # 입력 C=1 로짓에 시그모이드 적용
                softmax=softmax
            )
            # HausdorffDTLoss(include_background=include_background)
        else:
            # BoundaryLoss를 쓰려면 GT를 SDM으로 변환해 라벨로 넣어야 함.
            # self.boundary = BoundaryLoss()  # 라벨 SDM 파이프라인 준비 시 사용
            raise NotImplementedError("BoundaryLoss path requires SDM ground truth; set use_boundary_loss=False or implement SDM labels.")

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,C,[D,]H,W)  (C=1 or 2)
        y:      (B,1,[D,]H,W)  (class index or binary mask)
              - MONAI DiceCE는 y가 class index(one-hot 아님)여도 처리 가능
        """
        # 메인 + FocalTversky + HausdorffDT
        L = self.dicece(logits, y)

        if self.lambda_ftv > 0:
            L = L + self.lambda_ftv * self.ftv(logits, y)

        if self.lambda_hddt > 0:
            # BoundaryLoss를 쓰는 경우, 여기서 logits → prob 변환/SDM 라벨 주입 로직 필요
            L = L + self.lambda_hddt * self.hddt(logits, y)

        if self.lambda_cl > 0:
            # clDice는 확률이 필요
            B, C = logits.shape[:2]
            if C == 1:
                prob_pred = torch.sigmoid(logits)
                # y를 (B,1,...) 이진으로
                if y.shape[1] != 1:  # 만약 class index 형태라면
                    y_bin = (y > 0.5).float()
                else:
                    y_bin = (y > 0.5).float()
            else:  # C >= 2
                prob = torch.softmax(logits, dim=1)
                prob_pred = prob[:, 1:2]  # 전경 확률
                if y.shape[1] == 1:
                    # class index -> 전경 이진
                    y_bin = (y == 1).float()
                else:
                    # one-hot 라벨일 경우 전경 채널 사용
                    y_bin = y[:, 1:2].float()

            L = L + self.lambda_cl * cldice_loss(prob_pred.clamp(0,1), y_bin.clamp(0,1), skel_iters=self.cl_skel_iters)

        return L