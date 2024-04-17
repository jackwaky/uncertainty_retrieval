import abc

import numpy as np
import torch
from tqdm import tqdm

from evaluators.metric_calculators import ValidationMetricsCalculator
from evaluators.utils import multiple_index_from_attribute_list
from utils.metrics import AverageMeterSet

import shutil
import os
class AbstractBaseEvaluator(abc.ABC):
    def __init__(self, models, dataloaders, configs, top_k=(1, 10, 50)):
        self.models = models
        self.test_samples_dataloader = dataloaders['samples']
        self.test_query_dataloader = dataloaders['query']
        self.top_k = top_k if type(top_k) is tuple else tuple([int(k) for k in top_k.split(",")])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.attribute_matching_matrix = None
        self.ref_matching_matrix = None
        self.configs = configs

    def evaluate(self, epoch, key, model_idx):
        self.model_idx = model_idx

        all_results = {}
        all_test_features, all_test_attributes = self.extract_test_features_and_attributes()
        all_original_query_features, all_composed_query_features, all_query_attributes, all_ref_attributes = \
            self.extract_query_features_and_attributes()

        # Make sure test_loader is not shuffled! Otherwise, this will be incorrect
        if self.attribute_matching_matrix is None:
            self.attribute_matching_matrix = self._calculate_attribute_matching_matrix(all_query_attributes,
                                                                                       all_test_attributes)
        if self.ref_matching_matrix is None:
            self.ref_matching_matrix = self._calculate_attribute_matching_matrix(all_ref_attributes,
                                                                                 all_test_attributes)

        recall_calculator = ValidationMetricsCalculator(all_original_query_features, all_composed_query_features,
                                                        all_test_features, self.attribute_matching_matrix,
                                                        self.ref_matching_matrix, self.top_k, self.configs)
        recall_results, true_indices = recall_calculator()
        all_results.update(recall_results)
        print(all_results)

        # If Evaluation mode & want to Visualize the retrieved images
        if self.configs['mode'] == 'eval' and self.configs['visualize']:
            for image_idx in true_indices[0].keys():
                self.save_retrieved_images(true_indices, all_ref_attributes, all_query_attributes, all_test_attributes, key, image_idx)

        return all_results, recall_calculator

    @abc.abstractmethod
    def _extract_image_features(self, images):
        raise NotImplementedError

    @abc.abstractmethod
    def _extract_original_and_composed_features(self, images, modifiers, len_modifiers):
        raise NotImplementedError

    def extract_test_features_and_attributes(self):
        """
        Returns: (1) torch.Tensor of all test features, with shape (N_test, Embed_size)
                (2) list of test attributes, Size = N_test
        """
        self._to_eval_mode()

        dataloader = tqdm(self.test_samples_dataloader)
        all_test_attributes = []
        all_test_features = []
        with torch.no_grad():
            for batch_idx, (test_images, test_attr) in enumerate(dataloader):
                batch_size = test_images.size(0)
                test_images = test_images.to(self.device)

                features = self._extract_image_features(test_images)
                features = features.view(batch_size, -1).cpu()

                all_test_features.extend(features)
                all_test_attributes.extend(test_attr)

        return torch.stack(all_test_features), all_test_attributes

    def extract_query_features_and_attributes(self):
        """
            Returns: (1) torch.Tensor of all query features, with shape (N_query, Embed_size)
                    (2) list of target attributes, Size = N_query
            """
        self._to_eval_mode()

        dataloader = tqdm(self.test_query_dataloader)
        all_target_attributes = []
        all_ref_attributes = []
        all_composed_query_features = []
        all_original_query_features = []

        with torch.no_grad():
            for batch_idx, (ref_images, ref_attribute, modifiers, target_attribute, len_modifiers, attn_mask) in enumerate(
                    dataloader):
                batch_size = ref_images.size(0)
                ref_images = ref_images.to(self.device)
                modifiers, len_modifiers = modifiers.to(self.device), len_modifiers.to(self.device)
                if attn_mask != None:
                    attn_mask = attn_mask.to(self.device)

                original_features, composed_features = \
                    self._extract_original_and_composed_features(ref_images, modifiers, len_modifiers, attn_mask, self.model_idx)
                original_features = original_features.view(batch_size, -1).cpu()
                composed_features = composed_features.view(batch_size, -1).cpu()
                all_original_query_features.extend(original_features)
                all_composed_query_features.extend(composed_features)
                all_target_attributes.extend(target_attribute)
                all_ref_attributes.extend(ref_attribute)
        return torch.stack(all_original_query_features), torch.stack(
            all_composed_query_features), all_target_attributes, all_ref_attributes

    def _to_eval_mode(self, keys=None):
        keys = keys if keys else self.models.keys()
        for key in keys:
            self.models[key].eval()

    def _calculate_recall_at_k(self, most_similar_idx, all_test_attributes, all_target_attributes):
        average_meter_set = AverageMeterSet()

        for k in self.top_k:
            k_most_similar_idx = most_similar_idx[:, :k]
            for i, row in enumerate(k_most_similar_idx):
                most_similar_sample_attributes = multiple_index_from_attribute_list(all_test_attributes, row)
                target_attribute = all_target_attributes[i]
                correct = 1 if target_attribute in most_similar_sample_attributes else 0
                average_meter_set.update('recall_@{}'.format(k), correct)
        recall_results = average_meter_set.averages()
        return recall_results

    def save_retrieved_images(self, true_indices, all_ref_attributes, all_query_attributes, all_test_attributes, key, image_idx):
        matched_ref_attributes = [all_ref_attributes[idx] for idx in true_indices[0][image_idx]]
        matched_target_attributes = [all_query_attributes[idx] for idx in true_indices[0][image_idx]]
        matched_test_attributes = [all_test_attributes[idx] for idx in true_indices[1][image_idx]]

        domain = key.split('_')[-1]
        image_file_path = f"./data/{self.configs['dataset']}/image_data/{domain}"
        source_image_path = f'{image_file_path}/{matched_ref_attributes[0]}.jpg'
        answer_image_path = f'{image_file_path}/{matched_target_attributes[0]}.jpg'
        retrieved_image_path_list = [f'{image_file_path}/{retrieved_attribute}.jpg' for retrieved_attribute in
                                     matched_test_attributes]

        save_path = f"./visualize_result/{self.configs['experiment_description']}/{image_idx}"
        source_image_save_path = save_path + f'/{domain}/source_image/'
        answer_image_save_path = save_path + f'/{domain}/answer_image/'
        retrieved_image_save_path = save_path + f'/{domain}/retrieved_images/'

        for directory in [save_path, source_image_save_path, answer_image_save_path, retrieved_image_save_path]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        shutil.copy(source_image_path, source_image_save_path)
        shutil.copy(answer_image_path, answer_image_save_path)
        for idx, retrieved_image_path in enumerate(retrieved_image_path_list):
            cur_attribute = retrieved_image_path.split('/')[-1]
            changed_name = f'{idx}_{cur_attribute}'
            cur_retrieved_image_save_path = os.path.join(retrieved_image_save_path, changed_name)
            shutil.copy(retrieved_image_path, cur_retrieved_image_save_path)

    @staticmethod
    def _calculate_attribute_matching_matrix(all_query_attributes, all_test_attributes):
        all_query_attributes, all_test_attributes = np.array(all_query_attributes).reshape((-1, 1)), \
                                                    np.array(all_test_attributes).reshape((1, -1))
        return all_test_attributes == all_query_attributes
