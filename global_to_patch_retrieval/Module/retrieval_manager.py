#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm

from conv_onet.Data.crop_space import CropSpace
from points_shape_detect.Method.trans import normalizePointArray

from global_to_patch_retrieval.Method.path import createFileFolder


class RetrievalManager(object):

    def __init__(self, shapenet_dataset_folder_path=None):
        self.shapenet_dataset_folder_path = None
        self.shapenet_model_file_path_list = []

        if shapenet_dataset_folder_path is not None:
            self.loadShapeNetDataset(shapenet_dataset_folder_path)
        return

    def loadShapeNetDataset(self, shapenet_dataset_folder_path):
        assert os.path.exists(shapenet_dataset_folder_path)
        self.shapenet_dataset_folder_path = shapenet_dataset_folder_path
        self.shapenet_model_file_path_list = []

        class_folder_name_list = os.listdir(shapenet_dataset_folder_path)
        print(class_folder_name_list)
        print(len(class_folder_name_list))
        return True

    def generateAllCADFeature(self, shapenet_feature_folder_path):
        return True

    def generateRetrievalResult(self, obb_info_folder_path):
        assert os.path.exists(obb_info_folder_path)
        return True
