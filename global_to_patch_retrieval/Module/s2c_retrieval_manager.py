#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import open3d as o3d

from noc_transform.Data.obb import OBB
from noc_transform.Module.transform_generator import TransformGenerator

from scan2cad_dataset_manage.Module.dataset_loader import DatasetLoader
from scan2cad_dataset_manage.Module.object_model_map_manager import ObjectModelMapManager

from global_to_patch_retrieval.Method.feature import getPointsFeature

from global_to_patch_retrieval.Module.retrieval_manager import RetrievalManager


class S2CRetrievalManager(RetrievalManager):

    def __init__(self,
                 scan2cad_dataset_folder_path,
                 scannet_dataset_folder_path,
                 shapenet_dataset_folder_path,
                 scannet_object_dataset_folder_path,
                 scan2cad_object_model_map_dataset_folder_path,
                 shapenet_feature_folder_path,
                 print_progress=False):
        self.dataset_loader = DatasetLoader(scan2cad_dataset_folder_path,
                                            scannet_dataset_folder_path,
                                            shapenet_dataset_folder_path)
        self.object_model_map_manager = ObjectModelMapManager(
            scannet_object_dataset_folder_path, shapenet_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path)
        self.transform_generator = TransformGenerator()
        self.uniform_feature_dict = None

        self.loadUniformFeature(shapenet_feature_folder_path)
        return

    def loadUniformFeature(self, shapenet_feature_folder_path):
        assert os.path.exists(shapenet_feature_folder_path)

        uniform_feature_file_path = shapenet_feature_folder_path + \
            "../uniform_feature/uniform_feature.pkl"
        if not os.path.exists(uniform_feature_file_path):
            self.generateUniformFeatureDict(shapenet_feature_folder_path,
                                            uniform_feature_file_path,
                                            print_progress)

        assert os.path.exists(uniform_feature_file_path)
        with open(uniform_feature_file_path, 'rb') as f:
            self.uniform_feature_dict = pickle.load(f)
        return True

    def generateSceneRetrievalResult(self, scannet_scene_name):
        object_filename_list = self.object_model_map_manager.getObjectFileNameList(
            scannet_scene_name)

        for object_filename in object_filename_list:
            shapenet_model_dict = self.object_model_map_manager.getShapeNetModelDict(
                scannet_scene_name, object_filename)

            scannet_object_file_path = shapenet_model_dict[
                'scannet_object_file_path']
            shapenet_model_file_path = shapenet_model_dict[
                'shapenet_model_file_path']
            trans_matrix = np.array(shapenet_model_dict['trans_matrix'])
            print(scannet_object_file_path)
            print(shapenet_model_file_path)
            print(trans_matrix)
            exit()
        return True
