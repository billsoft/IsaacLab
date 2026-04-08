"""
占用损失函数

CE + Lovasz-Softmax (语义分类)
+ Smooth-L1 回归 (flow / orientation / angular_vel)，仅在 flow_mask=1 的体素上计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OccupancyLoss(nn.Module):
    def __init__(self, num_classes=18, ignore_index=255, lovasz_weight=0.5,
                 flow_weight=0.5, orient_weight=0.2, angvel_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lovasz_weight = lovasz_weight
        self.flow_weight = flow_weight
        self.orient_weight = orient_weight
        self.angvel_weight = angvel_weight

    def forward(self, outputs, sem_target, flow_target=None, flow_mask=None,
                orient_target=None, angvel_target=None):
        """
        outputs:       dict 来自 VoxelHead，包含 'semantic' 及可选回归字段
        sem_target:    [B, X, Y, Z]  uint8 语义类别 id
        flow_target:   [B, 2, X, Y, Z] float  (vx, vy)
        flow_mask:     [B, X, Y, Z]  uint8   1=动态体素
        orient_target: [B, 1, X, Y, Z] float 航向角
        angvel_target: [B, 1, X, Y, Z] float ωz
        """
        pred = outputs['semantic']
        B, C, X, Y, Z = pred.shape
        pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
        target_flat = sem_target.reshape(-1)

        valid_mask = target_flat != self.ignore_index
        pred_valid = pred_flat[valid_mask]
        target_valid = target_flat[valid_mask]

        zero = torch.tensor(0.0, device=pred.device)

        if pred_valid.numel() == 0:
            return {'total': zero, 'ce': zero, 'lovasz': zero,
                    'flow': zero, 'orient': zero, 'angvel': zero}

        ce_loss = F.cross_entropy(pred_valid, target_valid)
        lovasz_loss = self._lovasz_softmax(pred, sem_target)
        total_loss = ce_loss + self.lovasz_weight * lovasz_loss

        result = {'ce': ce_loss, 'lovasz': lovasz_loss,
                  'flow': zero, 'orient': zero, 'angvel': zero}

        # 回归损失（仅在 flow_mask=1 的动态体素上计算）
        if flow_mask is not None:
            dyn = flow_mask.bool()  # [B, X, Y, Z]

            if flow_target is not None and 'flow' in outputs and dyn.any():
                pred_flow = outputs['flow']           # [B, 2, X, Y, Z]
                gt_flow = flow_target.float()
                # 对每个动态体素计算 smooth_l1
                mask5 = dyn.unsqueeze(1).expand_as(pred_flow)
                fl = F.smooth_l1_loss(pred_flow[mask5], gt_flow[mask5])
                result['flow'] = fl
                total_loss = total_loss + self.flow_weight * fl

            if orient_target is not None and 'orientation' in outputs and dyn.any():
                pred_o = outputs['orientation']       # [B, 1, X, Y, Z]
                gt_o = orient_target.float()
                mask5 = dyn.unsqueeze(1).expand_as(pred_o)
                ol = F.smooth_l1_loss(pred_o[mask5], gt_o[mask5])
                result['orient'] = ol
                total_loss = total_loss + self.orient_weight * ol

            if angvel_target is not None and 'angular_vel' in outputs and dyn.any():
                pred_a = outputs['angular_vel']       # [B, 1, X, Y, Z]
                gt_a = angvel_target.float()
                mask5 = dyn.unsqueeze(1).expand_as(pred_a)
                al = F.smooth_l1_loss(pred_a[mask5], gt_a[mask5])
                result['angvel'] = al
                total_loss = total_loss + self.angvel_weight * al

        result['total'] = total_loss
        return result

    def _lovasz_softmax(self, pred, target):
        """Lovasz-Softmax, 强制 FP32 避免 AMP FP16 下 softmax+sort 精度损失"""
        with torch.amp.autocast('cuda', enabled=False):
            return self._lovasz_softmax_fp32(pred.float(), target)

    def _lovasz_softmax_fp32(self, pred, target):
        B, C, X, Y, Z = pred.shape
        prob = F.softmax(pred, dim=1)
        prob_flat = prob.permute(1, 0, 2, 3, 4).reshape(C, -1)
        target_flat = target.reshape(-1)

        valid_mask = target_flat != self.ignore_index
        prob_flat = prob_flat[:, valid_mask]
        target_flat = target_flat[valid_mask]

        if target_flat.numel() == 0:
            return torch.tensor(0.0, device=pred.device)

        # 只遍历实际存在的类 (避免对空类做无用排序)
        present_classes = target_flat.unique()
        loss_per_class = []
        for c in present_classes:
            if c == self.ignore_index:
                continue
            fg = (target_flat == c).float()
            errors = (fg - prob_flat[c]).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            loss_per_class.append((errors_sorted * grad).sum())

        if len(loss_per_class) == 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.stack(loss_per_class).mean()

    def _lovasz_grad(self, gt_sorted):
        n = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.cumsum(0)
        union = gts + (1 - gt_sorted).cumsum(0)
        jaccard = 1.0 - intersection / (union + 1e-6)
        if n > 1:
            jaccard[1:] = jaccard[1:] - jaccard[:-1]
        return jaccard
