#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import open3d as o3d
from tqdm import tqdm

from global_to_patch_retrieval.Method.feature import generateAllCADFeature
from global_to_patch_retrieval.Method.path import createFileFolder, renameFile


class RetrievalManager(object):

    def __init__(self,
                 shapenet_dataset_folder_path=None,
                 print_progress=False):
        self.shapenet_dataset_folder_path = None
        self.shapenet_model_file_path_list = []

        if shapenet_dataset_folder_path is not None:
            self.loadShapeNetDataset(shapenet_dataset_folder_path,
                                     print_progress)
        return

    def loadShapeNetDataset(self,
                            shapenet_dataset_folder_path,
                            print_progress=False):
        assert os.path.exists(shapenet_dataset_folder_path)
        self.shapenet_dataset_folder_path = shapenet_dataset_folder_path
        self.shapenet_model_file_path_list = []

        path_file = shapenet_dataset_folder_path + "../../tmp/paths.txt"

        if os.path.exists(path_file):
            with open(path_file, 'r') as f:
                lines = f.readlines()
                for_data = lines
                if print_progress:
                    print("[INFO][RetrievalManager::loadShapeNetDataset]")
                    print("\t start quick load shapenet model file paths...")
                    for_data = tqdm(for_data)
                for line in for_data:
                    self.shapenet_model_file_path_list.append(
                        line.split("\n")[0])
        else:
            tmp_path_file = path_file[:-4] + "_tmp.txt"
            createFileFolder(tmp_path_file)

            synset_folder_path_list = []

            synset_name_list = os.listdir(shapenet_dataset_folder_path)
            for synset_name in synset_name_list:
                synset_folder_path = shapenet_dataset_folder_path + synset_name + "/"
                if not os.path.isdir(synset_folder_path):
                    continue
                synset_folder_path_list.append(synset_folder_path)

            for_data = synset_folder_path_list
            if print_progress:
                print("[INFO][RetrievalManager::loadShapeNetDataset]")
                print("\t start load shapenet model file paths...")
                for_data = tqdm(for_data)
            for synset_folder_path in for_data:
                model_id_list = os.listdir(synset_folder_path)
                for model_id in model_id_list:
                    model_file_path = synset_folder_path + model_id + "/models/model_normalized.obj"
                    if not os.path.exists(model_file_path):
                        continue
                    self.shapenet_model_file_path_list.append(model_file_path)

            with open(tmp_path_file, 'w') as f:
                for_data = self.shapenet_model_file_path_list
                if print_progress:
                    print("[INFO][RetrievalManager::loadShapeNetDataset]")
                    print("\t start save shapenet model file paths...")
                    for_data = tqdm(for_data)
                for shapenet_model_file_path in for_data:
                    f.write(shapenet_model_file_path + "\n")

            renameFile(tmp_path_file, path_file)
        return True

    def generateUniformFeatureDict(self,
                                   shapenet_feature_folder_path,
                                   print_progress=False):
        assert os.path.exists(shapenet_feature_folder_path)

        feature_list = []
        mask_list = []

        for_data = self.shapenet_model_file_path_list
        if print_progress:
            print("[INFO][RetrievalManager::generateRetrievalResult]")
            print("\t start load shapenet model CAD features...")
            for_data = tqdm(for_data)
        for shapenet_model_file_path in for_data:
            model_label = shapenet_model_file_path.split("ShapeNetCore.v2/")[
                1].split("/models/model_normalized.obj")[0].replace("/", "_")
            feature_file_path = shapenet_feature_folder_path + model_label + ".pkl"

            assert os.path.exists(feature_file_path)

            with open(feature_file_path, 'rb') as f:
                feature_dict = pickle.load(f)

            feature_list.append(feature_dict['feature'])
            mask_list.append(feature_dict['mask'])

        uniform_feature_file_path = shapenet_feature_folder_path + "uniform_feature.pkl"

        tmp_uniform_feature_file_path = uniform_feature_file_path[:-4] + "_tmp.pkl"
        createFileFolder(tmp_uniform_feature_file_path)

        feature_array = np.array(feature_list)
        mask_array = np.array(mask_list)

        uniform_feature_dict = {
            'shapenet_model_file_path_list':
            self.shapenet_model_file_path_list,
            'feature_array': feature_array,
            'mask_array': mask_array
        }

        with open(tmp_uniform_feature_file_path, 'wb') as f:
            pickle.dump(uniform_feature_dict, f)
        return True

    def generateAllCADFeature(self,
                              shapenet_feature_folder_path,
                              print_progress=False):
        generateAllCADFeature(self.shapenet_model_file_path_list,
                              shapenet_feature_folder_path, print_progress)

        self.generateUniformFeatureDict(shapenet_feature_folder_path,
                                        print_progress)
        return True

    def generateRetrievalResult(self,
                                obb_info_folder_path,
                                shapenet_feature_folder_path,
                                print_progress=False):
        assert os.path.exists(obb_info_folder_path)
        assert os.path.exists(shapenet_feature_folder_path)

        uniform_feature_file_path = shapenet_feature_folder_path + "uniform_feature.pkl"

        if not os.path.exists(uniform_feature_file_path):
            self.generateUniformFeatureDict(shapenet_feature_folder_path,
                                            print_progress)

        assert os.path.exists(uniform_feature_file_path)
        with open(uniform_feature_file_path, 'rb') as f:
            uniform_feature_dict = pickle.load(f)
        return True
