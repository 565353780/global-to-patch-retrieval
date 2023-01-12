#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import pickle

import numpy as np
import open3d as o3d
from tqdm import tqdm

from noc_transform.Method.transform import transPoints
from points_shape_detect.Method.trans import normalizePointArray
from points_shape_detect.Method.matrix import getRotateMatrix

from global_to_patch_retrieval.Method.feature import (generateAllCADFeature,
                                                      getPointsFeature)
from global_to_patch_retrieval.Method.path import createFileFolder, renameFile
from global_to_patch_retrieval.Method.render import renderRetrievalResult


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
                                   save_uniform_feature_file_path,
                                   print_progress=False):
        assert os.path.exists(shapenet_feature_folder_path)

        feature_list = []
        mask_list = []

        for_data = self.shapenet_model_file_path_list
        if print_progress:
            print("[INFO][RetrievalManager::generateUniformFeatureDict]")
            print("\t start load shapenet model CAD features...")
            for_data = tqdm(for_data)
        for shapenet_model_file_path in for_data:
            model_label = shapenet_model_file_path.split("ShapeNetCore.v2/")[
                1].split("/models/model_normalized.obj")[0].replace("/", "_")
            feature_file_path = shapenet_feature_folder_path + model_label + ".pkl"

            if not os.path.exists(feature_file_path):
                break

            assert os.path.exists(feature_file_path)

            with open(feature_file_path, 'rb') as f:
                feature_dict = pickle.load(f)

            feature_list.append(feature_dict['feature'])
            mask_list.append(feature_dict['mask'])

        tmp_save_uniform_feature_file_path = save_uniform_feature_file_path[:
                                                                            -4] + "_tmp.pkl"
        createFileFolder(tmp_save_uniform_feature_file_path)

        feature_array = np.array(feature_list)
        mask_array = np.array(mask_list)

        uniform_feature_dict = {
            'shapenet_model_file_path_list':
            self.shapenet_model_file_path_list,
            'feature_array': feature_array,
            'mask_array': mask_array
        }

        with open(tmp_save_uniform_feature_file_path, 'wb') as f:
            pickle.dump(uniform_feature_dict, f)

        renameFile(tmp_save_uniform_feature_file_path,
                   save_uniform_feature_file_path)
        return True

    def generateAllCADFeature(self,
                              shapenet_feature_folder_path,
                              print_progress=False):
        generateAllCADFeature(self.shapenet_model_file_path_list,
                              shapenet_feature_folder_path, print_progress)

        self.generateUniformFeatureDict(shapenet_feature_folder_path,
                                        print_progress)
        return True

    def getObjectFeature(self, obb_info_folder_path, object_label):
        assert os.path.exists(obb_info_folder_path)
        object_file_path = obb_info_folder_path + "object/" + object_label + ".pcd"
        obb_trans_matrix_file_path = obb_info_folder_path + "obb_trans_matrix/" + object_label + ".json"
        assert os.path.exists(object_file_path)
        assert os.path.exists(obb_trans_matrix_file_path)

        pcd = o3d.io.read_point_cloud(object_file_path)

        with open(obb_trans_matrix_file_path, 'r') as f:
            obb_trans_matrix_dict = json.load(f)
        noc_trans_matrix = np.array(obb_trans_matrix_dict['noc_trans_matrix'])

        points = np.array(pcd.points)
        points = transPoints(points, noc_trans_matrix)
        rotate_matrix = getRotateMatrix([90, 0, -90], True)
        points = points @ rotate_matrix
        pcd.points = o3d.utility.Vector3dVector(points)

        points = np.array(pcd.points)

        object_feature, object_mask = getPointsFeature(points, False)
        return object_feature, object_mask

    def getAllObjectFeature(self, obb_info_folder_path, print_progress=False):
        object_feature_list = []
        object_mask_list = []

        object_folder_path = obb_info_folder_path + "object/"

        object_pcd_filename_list = os.listdir(object_folder_path)
        for_data = object_pcd_filename_list
        if print_progress:
            print("[INFO][RetrievalManager::getAllObjectFeature]")
            print("\t start get object features...")
            for_data = tqdm(for_data)
        for object_pcd_filename in for_data:
            if object_pcd_filename[-4:] != ".pcd":
                continue
            object_label = object_pcd_filename.split(".pcd")[0]

            object_feature, object_mask = self.getObjectFeature(
                obb_info_folder_path, object_label)

            object_feature_list.append(object_feature)
            object_mask_list.append(object_mask)

        object_feature_array = np.array(object_feature_list)
        object_mask_array = np.array(object_mask_list)
        return object_feature_array, object_mask_array

    def getObjectRetrievalResult(self,
                                 object_source_feature,
                                 object_mask,
                                 cad_feature_array,
                                 cad_mask_array,
                                 cad_model_file_path_list,
                                 print_progress=False):
        error_list = []
        for_data = range(cad_feature_array.shape[0])
        if print_progress:
            print("[INFO][RetrievalManager::getObjectRetrievalResult]")
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
                merge_error += np.linalg.norm(cad_feature - object_feature,
                                              ord=2)

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
                object_weight = object_only_feature_idx.shape[
                    0] / object_valid_num
            else:
                object_weight = 0
            object_error *= object_weight

            cad_error = 0

            cad_only_mask = cad_mask & ~object_mask
            cad_only_feature_idx = np.dstack(
                np.where(cad_only_mask == True))[0]
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

    def generateRetrievalResult(self,
                                obb_info_folder_path,
                                shapenet_feature_folder_path,
                                render=False,
                                print_progress=False):
        assert os.path.exists(shapenet_feature_folder_path)

        uniform_feature_file_path = shapenet_feature_folder_path + \
            "../uniform_feature/uniform_feature.pkl"
        if not os.path.exists(uniform_feature_file_path):
            self.generateUniformFeatureDict(shapenet_feature_folder_path,
                                            uniform_feature_file_path,
                                            print_progress)

        assert os.path.exists(uniform_feature_file_path)
        with open(uniform_feature_file_path, 'rb') as f:
            uniform_feature_dict = pickle.load(f)

        cad_file_path_list = uniform_feature_dict[
            'shapenet_model_file_path_list']
        cad_feature_array = uniform_feature_dict['feature_array']
        cad_mask_array = uniform_feature_dict['mask_array']

        object_feature_array, object_mask_array = self.getAllObjectFeature(
            obb_info_folder_path, print_progress)

        save_cad_model_folder_path = "/home/chli/chLi/auto-scan2cad/1314/CAD/"
        os.makedirs(save_cad_model_folder_path, exist_ok=True)

        retrieval_cad_model_file_path_list = []

        #  renderRetrievalResult(obb_info_folder_path,
        #  retrieval_cad_model_file_path_list, False)

        obb_trans_matrix_list = []

        object_folder_path = obb_info_folder_path + "object/"
        object_pcd_filename_list = os.listdir(object_folder_path)

        for object_pcd_filename in object_pcd_filename_list:
            if object_pcd_filename[-4:] != ".pcd":
                continue
            object_label = object_pcd_filename.split(".pcd")[0]

            obb_trans_matrix_file_path = obb_info_folder_path + "obb_trans_matrix/" + object_label + ".json"
            assert os.path.exists(obb_trans_matrix_file_path)

            with open(obb_trans_matrix_file_path, 'r') as f:
                obb_trans_matrix_dict = json.load(f)
            noc_trans_matrix = np.array(
                obb_trans_matrix_dict['noc_trans_matrix'])
            trans_matrix = np.linalg.inv(noc_trans_matrix)
            obb_trans_matrix_list.append(trans_matrix)

        for i in range(object_feature_array.shape[0]):
            save_cad_model_file_path = save_cad_model_folder_path + str(
                i) + ".ply"

            if os.path.exists(save_cad_model_file_path):
                retrieval_cad_model_file_path_list.append(
                    save_cad_model_file_path)
                continue

            obb_trans_matrix = obb_trans_matrix_list[i]
            object_feature = object_feature_array[i]
            object_mask = object_mask_array[i]

            cad_model_file_path = self.getObjectRetrievalResult(
                object_feature, object_mask, cad_feature_array, cad_mask_array,
                cad_file_path_list, print_progress)

            cad_mesh = o3d.io.read_triangle_mesh(cad_model_file_path)
            points = np.array(cad_mesh.vertices)
            points = normalizePointArray(points)
            rotate_matrix = getRotateMatrix([-90, 0, 90], False)
            points = points @ rotate_matrix
            points = transPoints(points, obb_trans_matrix)
            cad_mesh.vertices = o3d.utility.Vector3dVector(points)

            cad_mesh.compute_triangle_normals()

            o3d.io.write_triangle_mesh(save_cad_model_file_path,
                                       cad_mesh,
                                       write_ascii=True)

            retrieval_cad_model_file_path_list.append(save_cad_model_file_path)

        if render:
            renderRetrievalResult(obb_info_folder_path,
                                  retrieval_cad_model_file_path_list, False)
        return True
