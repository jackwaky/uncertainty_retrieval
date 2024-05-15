import random

import torch
import numpy as np
import wandb
import os
import json

from utils.metrics import AverageMeterSet

class ValidationMetricsCalculator:
    def __init__(self, original_query_features: torch.tensor, composed_query_features: torch.tensor,
                 test_features: torch.tensor, attribute_matching_matrix: np.array,
                 ref_attribute_matching_matrix: np.array, top_k: tuple, configs, key):
        self.original_query_features = original_query_features
        self.composed_query_features = composed_query_features
        self.test_features = test_features
        self.top_k = top_k
        self.attribute_matching_matrix = attribute_matching_matrix
        self.ref_attribute_matching_matrix = ref_attribute_matching_matrix
        self.num_query_features = composed_query_features.size(0)
        self.num_test_features = test_features.size(0)
        self.similarity_matrix = torch.zeros(self.num_query_features, self.num_test_features)
        self.top_scores = torch.zeros(self.num_query_features, max(top_k))
        self.most_similar_idx = torch.zeros(self.num_query_features, max(top_k))
        self.recall_results = {}
        self.recall_positive_queries_idxs = {k: [] for k in top_k}
        self.recall_negative_queries_idxs = {k: [] for k in top_k}
        self.similarity_matrix_calculated = False
        self.top_scores_calculated = False

        self.test_test_similarity_matrix_calculated = False

        self.configs = configs
        self.key = key
        self.domain = key.split('_')[-1]

    def __call__(self):
        self._calculate_similarity_matrix()
        self._calculate_test_diversity_matrix()
        # Filter query_feat == target_feat
        assert self.similarity_matrix.shape == self.ref_attribute_matching_matrix.shape
        self.similarity_matrix[self.ref_attribute_matching_matrix == True] = self.similarity_matrix.min() # make ref images in test set to min so that it could not be selected as a target image?
        return self._calculate_recall_at_k()

    def _calculate_similarity_matrix(self) -> torch.tensor:
        """
        query_features = torch.tensor. Size = (N_test_query, Embed_size)
        test_features = torch.tensor. Size = (N_test_dataset, Embed_size)
        output = torch.tensor, similarity matrix. Size = (N_test_query, N_test_dataset)
        """
        if not self.similarity_matrix_calculated:
            self.similarity_matrix = self.composed_query_features.mm(self.test_features.t())
            self.similarity_matrix_calculated = True

    def _calculate_test_diversity_matrix(self) -> torch.tensor:

        norm_test_features = self.test_features / self.test_features.norm(dim=1, keepdim=True)
        self.test_diversity_matrix = ( 1 - norm_test_features.mm(norm_test_features.t())) / 2

    def _calculate_recall_at_k(self):
        average_meter_set = AverageMeterSet()
        self.top_scores, self.most_similar_idx = self.similarity_matrix.topk(max(self.top_k))
        self.top_scores_calculated = True
        topk_attribute_matching = np.take_along_axis(self.attribute_matching_matrix, self.most_similar_idx.numpy(),
                                                     axis=1)

        true_indices_ref, true_indices_test = {}, {}
        unmatched_idx = {k: [] for k in self.top_k}
        for k in self.top_k:
            query_matched_vector = topk_attribute_matching[:, :k].sum(axis=1).astype(bool)
            self.recall_positive_queries_idxs[k] = [int(i) for i in np.where(query_matched_vector > 0)[0]]
            self.recall_negative_queries_idxs[k] = [int(i) for i in np.where(query_matched_vector == 0)[0]]

            if self.configs["mode"] == 'eval' and self.configs["log"]:
                prev_matched_idx = self.load_idx(k, 'matched')
                cur_matched_idx = self.recall_positive_queries_idxs[k]
                cur_unmatched_idx = self.recall_negative_queries_idxs[k]

                # # To compute difference between CosMo and MGUR
                # intersection = list(set(prev_matched_idx) & set(cur_unmatched_idx))

                unmatched_idx[k] = cur_matched_idx

            num_correct = query_matched_vector.sum()
            num_samples = len(query_matched_vector)
            average_meter_set.update('recall_@{}'.format(k), num_correct, n=num_samples)

            # Mean Average Precision
            precision_values = []
            diverse_precision_values = []
            for query_idx in range(num_samples):
                average_precision_at_k = 0
                diverse_precision_at_k = 0

                num_correct_per_query = topk_attribute_matching[query_idx, :k].sum().astype(bool).sum()
                if num_correct_per_query == 1:
                    retrieved_order =  np.where(topk_attribute_matching[query_idx, :k])[0][0] + 1
                    average_precision_at_k = num_correct_per_query / retrieved_order

                    retrieved_indices = self.most_similar_idx[query_idx, :retrieved_order].tolist()
                    assert(len(retrieved_indices)==retrieved_order)

                    candidate_scores = self.test_diversity_matrix[retrieved_indices[-1], retrieved_indices[:-1]].tolist()
                    diverse_precision_at_k = min(candidate_scores) if retrieved_order != 1 else 1

                precision_values.append(average_precision_at_k)
                diverse_precision_values.append(diverse_precision_at_k)

            average_meter_set.update('map_@{}'.format(k), sum(precision_values), n=num_samples)
            average_meter_set.update('mdap_@{}'.format(k), sum(diverse_precision_values), n=num_samples)

            if self.configs['mode'] == 'eval' and self.configs['visualize'] and k == 10:
                # visualized_idx = self.configs['visualized_image_idx']
                # visualized_idx = unmatched_idx[10]
                # visualized_idx = self.load_idx(10, 'wrong')
                visualized_idx = os.listdir(f"/home/jaehyun98/git/uncertain_retrieval/visualize_result/MGUR_eval/{self.domain}/intersection")
                for idx in visualized_idx:
                    idx = int(idx)
                    true_indices_ref[idx] = [idx]
                    matched_matrix = self.similarity_matrix[[idx]]
                    true_indices_test[idx] = matched_matrix.topk(10, dim=1)[1].tolist()[0]

        if self.configs["mode"] == 'eval':
            # save unmatched_idx
            if self.configs["ambiguity"]:
                self.save_idx(unmatched_idx, 'unmatched')
            # save matched_idx
            else:
                self.save_idx(self.recall_positive_queries_idxs, 'matched')
                self.save_idx(self.recall_negative_queries_idxs, 'wrong')

        recall_results = average_meter_set.averages()
        return recall_results, [true_indices_ref, true_indices_test]

    def save_idx(self, indexes, prefix):
        domain = self.key.split('_')[-1]
        save_path = f"./visualize_result/{prefix}_{self.configs['experiment_description']}/{domain}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(f'{save_path}/log.json', 'w') as f:
            json.dump(indexes, f, indent=4)

    def load_idx(self, k, is_matched):
        domain = self.key.split('_')[-1]
        # file_path = f"./visualize_result/{is_matched}_{self.configs['experiment_description']}/{domain}/log.json"
        file_path = f"./visualize_result/matched_CosMo_eval/{domain}/log.json"

        with open(file_path, 'r') as f:
            json_file = json.load(f)

        cur_k_matched_idx = json_file[str(k)]
        return cur_k_matched_idx


    @staticmethod
    def _multiple_index_from_attribute_list(attribute_list, indices):
        attributes = []
        for idx in indices:
            attributes.append(attribute_list[idx.item()])
        return attributes
