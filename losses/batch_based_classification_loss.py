import torch
import torch.nn.functional as F

from trainers.abc import AbstractBaseMetricLoss


class BatchBasedClassificationLoss(AbstractBaseMetricLoss):
    def __init__(self):
        super().__init__()

    def forward(self, ref_features, tar_features, batch_idx):
        return self.cal_loss(ref_features, tar_features, batch_idx)

    @classmethod
    def cal_loss(cls, ref_features, tar_features, batch_idx):
        batch_size = ref_features.size(0)
        device = ref_features.device

        pred = ref_features.mm(tar_features.transpose(0, 1))
        loss_ls = []
        if batch_idx == None:
            labels = torch.arange(0, batch_size).long().to(device)
        # Calculate the min loss by put whole batch
        elif batch_idx == -1:
            labels = torch.arange(0, batch_size).long().to(device)
            for idx in range(batch_size):
                cur_pred, cur_label = pred[idx, :], labels[idx]
                cur_loss = F.cross_entropy(cur_pred, cur_label)
                loss_ls.append(cur_loss)
        else:
            labels = torch.tensor([batch_idx]).long().to(device)
        return F.cross_entropy(pred, labels)

    @classmethod
    def code(cls):
        return 'batch_based_classification_loss'
