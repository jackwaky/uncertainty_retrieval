import torch
import torch.nn.functional as F

from trainers.abc import AbstractBaseMetricLoss


class BiBatchBasedClassificationLoss(AbstractBaseMetricLoss):
    def __init__(self):
        super().__init__()

    def forward(self, ref_features, tar_features, rev_ref_features, rev_tar_features):
        return self.cal_loss(ref_features, tar_features, rev_ref_features, rev_tar_features)

    @classmethod
    def cal_loss(cls, ref_features, tar_features, rev_ref_features, rev_tar_features):
        batch_size = ref_features.size(0)
        device = ref_features.device

        pred = ref_features.mm(tar_features.transpose(0, 1))
        labels = torch.arange(0, batch_size).long().to(device)

        rev_pred = rev_ref_features.mm(rev_tar_features.transpose(0, 1))
        loss_forward = F.cross_entropy(pred, labels)
        loss_rev = F.cross_entropy(rev_pred, labels)
        loss = loss_forward + .5 * loss_rev
        return loss

    @classmethod
    def code(cls):
        return 'bi_batch_based_classification_loss'
