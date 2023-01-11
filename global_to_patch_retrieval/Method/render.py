#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import open3d as o3d

from noc_transform.Method.transform import transPoints
from points_shape_detect.Method.trans import normalizePointArray


def renderRetrievalResult(obb_info_folder_path,
                          retrieval_cad_model_file_path_list):
    assert os.path.exists(obb_info_folder_path)

    object_folder_path = obb_info_folder_path + "object/"
    object_pcd_filename_list = os.listdir(object_folder_path)

    object_pcd_list = []
    obb_trans_matrix_list = []

    for object_pcd_filename in object_pcd_filename_list:
        if object_pcd_filename[-4:] != ".pcd":
            continue
        object_label = object_pcd_filename.split(".pcd")[0]

        object_file_path = obb_info_folder_path + "object/" + object_label + ".pcd"
        obb_trans_matrix_file_path = obb_info_folder_path + "obb_trans_matrix/" + object_label + ".json"
        assert os.path.exists(object_file_path)
        assert os.path.exists(obb_trans_matrix_file_path)

        object_pcd = o3d.io.read_point_cloud(object_file_path)
        object_pcd_list.append(object_pcd)

        with open(obb_trans_matrix_file_path, 'r') as f:
            obb_trans_matrix_dict = json.load(f)
        noc_trans_matrix = np.array(obb_trans_matrix_dict['noc_trans_matrix'])
        trans_matrix = np.linalg.inv(noc_trans_matrix)
        obb_trans_matrix_list.append(trans_matrix)

    render_list = []

    for i in range(len(retrieval_cad_model_file_path_list)):
        object_pcd = object_pcd_list[i]
        obb_trans_matrix = obb_trans_matrix_list[i]
        cad_model_file_path = retrieval_cad_model_file_path_list[i]
        cad_mesh = o3d.io.read_triangle_mesh(cad_model_file_path)

        points = np.array(cad_mesh.vertices)
        points = normalizePointArray(points)
        points = transPoints(points, obb_trans_matrix)
        cad_mesh.vertices = o3d.utility.Vector3dVector(points)

        render_list.append(object_pcd)
        render_list.append(cad_mesh)

    o3d.visualization.draw_geometries(render_list)
    return True
