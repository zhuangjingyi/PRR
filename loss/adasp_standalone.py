"""
AdaSP Loss - Standalone version, ready for use in any PyTorch project.

Paper: Adaptive Sparse Pairwise Loss for Object Re-Identification (CVPR 2023)
Source: https://github.com/zhouxiaotu/AdaSP

Usage:
    from adasp_standalone import AdaSPLoss

    loss_fn = AdaSPLoss(temp=0.05, loss_type='adasp')  # or 'sp-h', 'sp-lh'
    features = model(images)  # shape: [B, D], already L2-normalized or not
    loss = loss_fn(features, labels)  # labels: [B] with P*K structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdaSPLoss(object):
    """
    Adaptive Sparse Pairwise (AdaSP) Loss.

    Three variants:
        - 'sp-h' : Sparse Pairwise with Hard-Hard mining
        - 'sp-lh': Sparse Pairwise with Hard-Easy mining
        - 'adasp': Adaptive fusion of HH and HE (recommended)

    Args:
        temp: Temperature factor, lower = harder. Default 0.05 works well.
        loss_type: One of {'adasp', 'sp-h', 'sp-lh'}.
    """

    def __init__(self, temp: float = 0.05, loss_type: str = 'adasp'):
        if loss_type not in ('adasp', 'sp-h', 'sp-lh'):
            raise ValueError(f"loss_type must be 'adasp', 'sp-h', or 'sp-lh', got '{loss_type}'")
        self.temp = temp
        self.loss_type = loss_type

    def __call__(self, feats: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats:   Feature embeddings, shape [B, D].
            targets: Class labels, shape [B]. Must follow PK sampling:
                     each class has exactly K = B // num_classes instances.

        Returns:
            Scalar loss.
        """
        device = feats.device

        # ---- L2 normalize features ----
        feats_n = F.normalize(feats, dim=1)

        B = feats_n.size(0)
        N_id = len(torch.unique(targets))
        N_ins = B // N_id  # instances per class (K in PK sampling)

        scale = 1.0 / self.temp

        # ---- scaled similarity matrix ----
        sim_qq = torch.matmul(feats_n, feats_n.T)
        sf_sim_qq = sim_qq * scale

        # ---- helper matrices (identity blocks) ----
        right_factor = torch.from_numpy(
            np.kron(np.eye(N_id), np.ones((N_ins, 1)))
        ).float().to(device)
        pos_mask = right_factor.clone()
        left_factor = torch.from_numpy(
            np.kron(np.eye(N_id), np.ones((1, N_ins)))
        ).float().to(device)

        # ============================================================
        #  Hard-Hard (HH) mining for positives
        # ============================================================
        mask_HH = torch.from_numpy(
            np.kron(np.eye(N_id), -1.0 * np.ones((N_ins, N_ins)))
        ).float().to(device)
        mask_HH[mask_HH == 0] = 1.0

        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))
        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id, device=device)
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH == 0] = 1.0
        pos_sim_HH = 1.0 / pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1 - pos_mask_id) + pos_sim_HH.mul(pos_mask_id)

        ID_sim_HH_L1 = F.normalize(ID_sim_HH, p=1, dim=1)

        # ============================================================
        #  Hard-Easy (HE) mining for positives
        # ============================================================
        mask_HE = torch.from_numpy(
            np.kron(np.eye(N_id), -1.0 * np.ones((N_ins, N_ins)))
        ).float().to(device)
        mask_HE[mask_HE == 0] = 1.0

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        pos_sim_HE[pos_sim_HE == 0] = 1.0
        pos_sim_HE = 1.0 / pos_sim_HE
        ID_sim_HE = ID_sim_HE.mul(1 - pos_mask) + pos_sim_HE.mul(pos_mask)

        # Apply hard-hard style for negatives
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        ID_sim_HE_L1 = F.normalize(ID_sim_HE, p=1, dim=1)

        # ============================================================
        #  Adaptive fusion (AdaSP only)
        # ============================================================
        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach() / scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach() / scale
        wt_l = 2 * weight_sim_HE * weight_sim_HH / (weight_sim_HH + weight_sim_HE + 1e-8)
        wt_l[weight_sim_HH < 0] = 0.0
        both_sim = l_sim * wt_l + s_sim * (1 - wt_l)

        adaptive_pos = torch.diag(torch.exp(both_sim))
        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mul(1 - pos_mask_id)
        adaptive_sim_mat_L1 = F.normalize(adaptive_sim_mat, p=1, dim=1)

        # ---- Loss computation ----
        loss_sph = -torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_splh = -torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adasp = -torch.log(torch.diag(adaptive_sim_mat_L1)).mean()

        if self.loss_type == 'sp-h':
            return loss_sph
        elif self.loss_type == 'sp-lh':
            return loss_splh
        else:
            return loss_adasp
