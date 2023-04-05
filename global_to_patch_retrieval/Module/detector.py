#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import open3d as o3d
import torch

from global_to_patch_retrieval.Dataset.scan2cad import Scan2CAD
from global_to_patch_retrieval.Method.device import toCpu, toCuda, toNumpy
from global_to_patch_retrieval.Model.retrieval_net import RetrievalNet


class Detector(object):

    def __init__(self, model_file_path=None):
        self.model = RetrievalNet(True).cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][Detector::loadModel]")
        print("\t start loading model from :")
        print("\t", model_file_path)
        model_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_dict['model'])
        return True

    def detectDataset(self):
        dataset_file = \
            "/home/chli/chLi/Scan-CAD Object Similarity Dataset/scan2cad_objects_split.json"
        scannet_folder = \
            "/home/chli/chLi/Scan-CAD Object Similarity Dataset/objects_aligned/"
        shapenet_folder = \
            "/home/chli/chLi/Scan-CAD Object Similarity Dataset/objects_aligned/"

        dataset = Scan2CAD(dataset_file, scannet_folder, shapenet_folder,
                           ["train"])

        for i in range(len(dataset)):
            data = dataset.__getitem__(i)
            for key, item in data['inputs'].items():
                print(key, '->', item.shape)
            return
        return True

    def detectSceneTrans(self, data):
        self.model.eval()

        toCuda(data)

        wall_position = data['inputs']['wall_position']
        floor_position = data['inputs']['floor_position']
        trans_object_obb = data['inputs']['trans_object_obb']

        wall_num = wall_position.shape[0]
        floor_num = floor_position.shape[0]
        object_num = trans_object_obb.shape[0]

        data['inputs']['floor_position'] = data['inputs']['floor_position'].to(
            torch.float32).reshape(1, floor_num, -1)
        data['inputs']['floor_normal'] = data['inputs']['floor_normal'].to(
            torch.float32).reshape(1, floor_num, -1)
        data['inputs']['floor_z_value'] = data['inputs']['floor_z_value'].to(
            torch.float32).reshape(1, floor_num, -1)

        data['inputs']['wall_position'] = data['inputs']['wall_position'].to(
            torch.float32).reshape(1, wall_num, -1)
        data['inputs']['wall_normal'] = data['inputs']['wall_normal'].to(
            torch.float32).reshape(1, wall_num, -1)

        data['inputs']['trans_object_obb'] = data['inputs'][
            'trans_object_obb'].to(torch.float32).reshape(1, object_num, -1)
        data['inputs']['trans_object_abb'] = data['inputs'][
            'trans_object_abb'].to(torch.float32).reshape(1, object_num, -1)
        data['inputs']['trans_object_obb_center'] = data['inputs'][
            'trans_object_obb_center'].to(torch.float32).reshape(
                1, object_num, -1)

        data['inputs']['trans_object_obb_center_dist'] = data['inputs'][
            'trans_object_obb_center_dist'].to(torch.float32).reshape(
                1, -1, 1)
        data['inputs']['trans_object_abb_eiou'] = data['inputs'][
            'trans_object_abb_eiou'].to(torch.float32).reshape(1, -1, 1)

        data = self.model(data)

        toCpu(data)
        toNumpy(data)
        return data
