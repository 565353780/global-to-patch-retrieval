#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from conv_onet.Data.crop_space import CropSpace
from points_shape_detect.Method.trans import normalizePointArray
from tqdm import tqdm

from global_to_patch_retrieval.Method.path import createFileFolder


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
                self.shapenet_model_file_path_list = f.readlines()
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

        print("model total :", len(self.shapenet_model_file_path_list))
        return True

    def generateAllCADFeature(self, shapenet_feature_folder_path):
        return True

    def generateRetrievalResult(self, obb_info_folder_path):
        assert os.path.exists(obb_info_folder_path)
        return True
