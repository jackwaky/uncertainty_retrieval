from tqdm import tqdm

from trainers.abc import AbstractBaseTrainer
from utils.metrics import AverageMeterSet

import torch
from .abc import LoggingService


class TIRGTrainer(AbstractBaseTrainer):
    def __init__(self, models, train_dataloader, criterions, optimizers, lr_schedulers, num_epochs,
                 train_loggers, val_loggers, evaluators, train_evaluators, configs, *args, **kwargs):
        super().__init__(models, train_dataloader, criterions, optimizers, lr_schedulers, num_epochs,
                         train_loggers, val_loggers, evaluators, train_evaluators, configs, *args, **kwargs)
        self.num_models = configs["num_models"]
        self.lower_image_encoder = models['lower_image_encoder']
        self.upper_image_encoder = models['upper_image_encoder']
        self.text_encoder = models['text_encoder']
        self.text_fc = models['text_fc'] if 'text_fc' in self.models else None
        self.compositor = [models[f'compositor_{model_idx}'] for model_idx in range(self.num_models)]
        self.augmenter = models['augmenter'] if 'augmenter' in self.models else None
        self.metric_loss = self.criterions['metric_loss']


        self.train_logging_service = LoggingService(train_loggers)

    def train_one_epoch(self, epoch):
        average_meter_set = {i : AverageMeterSet() for i in range(self.num_models)}
        # average_meter_set = AverageMeterSet()
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
                tar_mid_features, _ = self.lower_image_encoder(tar_images)
                tar_features = self.upper_image_encoder(tar_mid_features)

                # Encode and Fuse Reference Images with Texts
                ref_mid_features, _ = self.lower_image_encoder(ref_images)
                if self.text_fc != None:
                    attn_mask = attn_mask.to(self.device)
                    text_features = self.text_encoder(modifiers, attn_mask)
                    text_features = self.text_fc(text_features)
                else:
                    text_features = self.text_encoder(modifiers, len_modifiers)

                composed_ref_features, _ = self.compositor[i](ref_mid_features, text_features)
                composed_ref_features = self.upper_image_encoder(composed_ref_features)

                # Add Gaussian noisy to feature and compute Loss
                if self.augmenter != None:
                    augmented_tar_features = self.augmenter(tar_features)
                    augmented_tar_feature_list.append(augmented_tar_features)

                composed_ref_feature_list.append(composed_ref_features)
                tar_feature_list.append(tar_features)

            for b in range(sample_size):
                with torch.no_grad():
                    loss_ls = [self.metric_loss(torch.unsqueeze(composed_ref_feature_list[j][b], dim=0), tar_feature_list[j], b) for j in range(self.num_models)]

                _, min_index_ls = torch.topk(-(torch.tensor(loss_ls)), 1)

                for index in min_index_ls:
                    assgn_ls[index].append(b)

            # with torch.no_grad():
            #      loss_ls = [self.metric_loss(composed_ref_feature_list[j], tar_feature_list[j], -1) for j in range(self.num_models)]
            #
            # for b in range(sample_size):
            #     cur_loss_ls = [loss_ls[model_idx][1][b] for model_idx in range(self.num_models)]
            #     _, min_index_ls = torch.topk(-(torch.tensor(cur_loss_ls)), 1)
            #
            #     for index in min_index_ls:
            #         assgn_ls[index].append(b)

            assgn_ls_log = {f"assign_model_{i}" : len(assgn_ls[i]) for i in range(len(assgn_ls))}
            self.train_logging_service.log(assgn_ls_log, step=batch_idx + epoch * len(train_dataloader), model_idx=0)

            for m, assign in enumerate(assgn_ls):
                if(len(assign)!=0):

                    if(len(assign)>1):
                        loss = self.metric_loss(composed_ref_feature_list[m][assign], tar_feature_list[m][assign], None)
                    else:
                        loss = self.metric_loss(torch.unsqueeze(composed_ref_feature_list[m][assign[0]], dim=0), torch.unsqueeze(tar_feature_list[m][assign[0]], dim=0), None)

                    loss.backward()
                    average_meter_set[m].update('loss', loss.item())
                    # if m == 0:
            self._update_grad()
                    # self._update_compositor_grad(model_idx=m)

            break


        # train_results = {i : average_meter_set[i].averages() for i in range(self.num_models)}
        # # optimizers_dict = self._get_state_dicts(self.optimizers)
        # # for model_idx in optimizers_dict.keys():
        # for model_idx in range(self.num_models):
        #     cur_optimizer_dict = self._get_state_dicts(self.optimizers[model_idx])
        #     for key in cur_optimizer_dict.keys():
        #         train_results[model_idx][key+'_lr'] = cur_optimizer_dict[key]["param_groups"][0]["lr"]
        # self._step_schedulers()

        train_results = {i : average_meter_set[i].averages() for i in range(self.num_models)}
        optimizers_dict = self._get_state_dicts(self.optimizers)
        for model_idx in range(self.num_models):
            for key in optimizers_dict.keys():
                train_results[model_idx][key + '_lr'] = optimizers_dict[key]["param_groups"][0]["lr"]
        self._step_schedulers()
        return train_results

    @classmethod
    def code(cls) -> str:
        return 'tirg'
