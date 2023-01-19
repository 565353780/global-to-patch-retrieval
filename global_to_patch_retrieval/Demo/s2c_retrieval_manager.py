#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../conv-onet")
sys.path.append("../noc-transform")

from global_to_patch_retrieval.Module.s2c_retrieval_manager import S2CRetrievalManager


def demo():
    scan2cad_dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
    scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
    shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
    scan2cad_object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"
    shapenet_feature_folder_path = "/home/chli/chLi/ShapeNet/features/"
    print_progress = True

    s2c_retrieval_manager = S2CRetrievalManager(
        scan2cad_dataset_folder_path, scannet_dataset_folder_path,
        shapenet_dataset_folder_path, scannet_object_dataset_folder_path,
        scan2cad_object_model_map_dataset_folder_path,
        shapenet_feature_folder_path, print_progress)

    valid_scene_name_list = [
        'scene0474_02', 'scene0000_01', 'scene0667_01', 'scene0500_00',
        'scene0247_01', 'scene0644_00'
    ]

    scannet_scene_name = valid_scene_name_list[1]

    s2c_retrieval_manager.generateSceneRetrievalResult(scannet_scene_name)
    return True