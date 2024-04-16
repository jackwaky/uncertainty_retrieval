from tqdm import tqdm

from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet

import torch


class TIRGTrainer(AbstractBaseTrainer):
    def __init__(self, models, train_dataloader, criterions, optimizers, lr_schedulers, num_epochs,
                 train_loggers, val_loggers, evaluators, *args, **kwargs):
        super().__init__(models, train_dataloader, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, evaluators, *args, **kwargs)
        self.lower_image_encoder = [model['lower_image_encoder'] for model in self.models]
        self.upper_image_encoder = [model['upper_image_encoder'] for model in self.models]
        self.text_encoder = [model['text_encoder'] for model in self.models]
        self.text_fc = [model['text_fc'] for model in self.models] if 'text_fc' in self.models[0] else None
        self.compositor = [model['layer4'] for model in self.models]
        self.augmenter = [model['augmenter'] for model in self.models] if 'augmenter' in self.models[0] else None
        self.metric_loss = self.criterions['metric_loss']
        self.num_models = len(self.models)

    def train_one_epoch(self, epoch):
        average_meter_set = AverageMeterSet()
        train_dataloader = tqdm(self.train_dataloader, desc="Epoch {}".format(epoch))

        for batch_idx, (ref_images, tar_images, modifiers, len_modifiers, attn_mask) in enumerate(train_dataloader):
            ref_images, tar_images = ref_images.to(self.device), tar_images.to(self.device)
            modifiers, len_modifiers = modifiers.to(self.device), len_modifiers.to(self.device)

            sample_size = ref_images.size()[0]
            assgn_ls = []

            for _ in range(self.num_models):
                assgn_ls.append([])

            self._reset_grad()
            # Get num_models number of features per ensemble model
            composed_ref_feature_list, tar_feature_list, augmented_tar_feature_list = [], [], []
            for i in range(self.num_models):
                # Encode Target Images
                tar_mid_features, _ = self.lower_image_encoder[i](tar_images)
                tar_features = self.upper_image_encoder[i](tar_mid_features)

                # Encode and Fuse Reference Images with Texts
                ref_mid_features, _ = self.lower_image_encoder[i](ref_images)
                if self.text_fc != None:
                    attn_mask = attn_mask.to(self.device)
                    text_features = self.text_encoder[i](modifiers, attn_mask)
                    text_features = self.text_fc[i](text_features)
                else:
                    text_features = self.text_encoder[i](modifiers, len_modifiers)

                composed_ref_features, _ = self.compositor[i](ref_mid_features, text_features)
                composed_ref_features = self.upper_image_encoder[i](composed_ref_features)

                # Add Gaussian noisy to feature and compute Loss
                if self.augmenter != None:
                    augmented_tar_features = self.augmenter[i](tar_features)
                    augmented_tar_feature_list.append(augmented_tar_features)

                composed_ref_feature_list.append(composed_ref_features)
                tar_feature_list.append(tar_features)

            for b in range(sample_size):
                with torch.no_grad():
                    loss_ls = [self.metric_loss(torch.unsqueeze(composed_ref_feature_list[j][b], dim=0), tar_feature_list[j], b) for j in range(self.num_models)]

                _, min_index_ls = torch.topk(-(torch.tensor(loss_ls)), 1)

                for index in min_index_ls:
                    assgn_ls[index].append(b)

            # print(assgn_ls)

            for m, assign in enumerate(assgn_ls):
                if(len(assign)!=0):

                    if(len(assign)>1):
                        loss = self.metric_loss(composed_ref_feature_list[m][assign], tar_feature_list[m][assign], None)
                    else:
                        loss = self.metric_loss(torch.unsqueeze(composed_ref_feature_list[m][assign[0]], dim=0), torch.unsqueeze(tar_feature_list[m][assign[0]], dim=0), None)

                    loss.backward()
                    average_meter_set.update('loss', loss.item())
                    self._update_grad(model_idx=m)

            # break


        train_results = average_meter_set.averages()
        optimizers_dict = self._get_state_dicts(self.optimizers)
        for model_idx in optimizers_dict.keys():
            for key in optimizers_dict[model_idx].keys():
                train_results[key+'_lr'] = optimizers_dict[model_idx][key]["param_groups"][0]["lr"]
        self._step_schedulers()
        return train_results

    @classmethod
    def code(cls) -> str:
        return 'tirg'
