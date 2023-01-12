#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm


def getObjectRetrievalResult(object_source_feature,
                             object_mask,
                             cad_feature_array,
                             cad_mask_array,
                             cad_model_file_path_list,
                             print_progress=False):
    error_list = []
    for_data = range(cad_feature_array.shape[0])
    if print_progress:
        print("[INFO][retrieval::getObjectRetrievalResult]")
        print("\t start get object retrieval result...")
        for_data = tqdm(for_data)
    for i in for_data:
        cad_source_feature = cad_feature_array[i]
        cad_mask = cad_mask_array[i]
        cad_valid_num = np.where(cad_mask == True)[0].shape[0]

        merge_mask = cad_mask & object_mask
        merge_feature_idx = np.dstack(np.where(merge_mask == True))[0]

        merge_error = 0

        for j, k, l in merge_feature_idx:
            cad_feature = cad_source_feature[j, k, l].flatten()
            object_feature = object_source_feature[j, k, l].flatten()
            merge_error += np.linalg.norm(cad_feature - object_feature, ord=2)

        if len(merge_feature_idx) > 0:
            merge_error /= len(merge_feature_idx)

        object_error = 0

        object_only_mask = ~cad_mask & object_mask
        object_only_feature_idx = np.dstack(
            np.where(object_only_mask == True))[0]
        for j, k, l in object_only_feature_idx:
            object_feature = object_source_feature[j, k, l].flatten()
            object_error += np.linalg.norm(object_feature, ord=2)

        object_valid_num = np.where(object_mask == True)[0].shape[0]
        if object_valid_num > 0:
            object_weight = object_only_feature_idx.shape[0] / object_valid_num
        else:
            object_weight = 0
        object_error *= object_weight

        cad_error = 0

        cad_only_mask = cad_mask & ~object_mask
        cad_only_feature_idx = np.dstack(np.where(cad_only_mask == True))[0]
        for j, k, l in cad_only_feature_idx:
            cad_feature = cad_source_feature[j, k, l].flatten()
            cad_error += np.linalg.norm(cad_feature, ord=2)

        if cad_valid_num > 0:
            cad_weight = cad_only_feature_idx.shape[0] / cad_valid_num
        else:
            cad_weight = 0
        cad_error *= cad_weight

        error = merge_error + 0.8 * object_error + 0.2 * cad_error

        error_list.append(error)

    min_error_idx = np.argmin(error_list)
    min_error_cad_model_file_path = cad_model_file_path_list[min_error_idx]
    return min_error_cad_model_file_path
