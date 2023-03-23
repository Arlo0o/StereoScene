#import open3d as o3d
import trimesh
import mmcv
import numpy as np
import torch
import yaml, os
import numba as nb

from mmdet.datasets.builder import PIPELINES
from torch.utils import data

import pdb

# create voxel-level labels from lidarseg points
@PIPELINES.register_module()
class CreateVoxelLabels(object):
    def __init__(self, point_cloud_range, grid_size, unoccupied=0):
        self.grid_size = np.array(grid_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.unoccupied = unoccupied
        
        # index=0: min_bound + voxel_size / 2 
        # index=-1: max_bound - voxel_size / 2
        self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.grid_size

    def __call__(self, results):
        # [N, 4] in (x, y, z, cls_id)
        points_seg = results['points_occ']
        # clip and convert to indices
        # e.g. points within [min_bound, min_bound + voxel_size] are classfied into index0
        eps = 0.01
        points_grid_ind = np.floor((np.clip(points_seg[:, :3], self.point_cloud_range[:3], 
                self.point_cloud_range[3:] - eps) - self.point_cloud_range[:3]) / self.voxel_size).astype(np.int)
        
        label_voxel_pair = np.concatenate([points_grid_ind, points_seg[:, -1:]], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((points_grid_ind[:, 0], points_grid_ind[:, 1], points_grid_ind[:, 2])), :]
        label_voxel_pair = label_voxel_pair.astype(np.int64)
        
        # 0: noise, 1-16 normal classes, 17 unoccupied
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
        processed_label = nb_process_label(processed_label, label_voxel_pair)
        results['gt_occ'] = processed_label
        
        return results

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

# modified from MonoScene, compute the relations for supervision
@PIPELINES.register_module()
class CreateRelationLabels(object):
    def __init__(self, point_cloud_range, grid_size, class_names, output_scale=2.0):
        self.grid_size = np.array(grid_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.grid_size
        self.scene_size = self.point_cloud_range[3:] - self.point_cloud_range[:3]
        
        self.class_names = class_names
        self.n_classes = len(class_names)
        
        self.output_scale = output_scale
        self.init_voxel_coordinates()
    
    def _downsample_label(self, label, downscale=4):
        r"""downsample the labeled data,
        code taken from https://github.com/waterljwant/SSC/blob/master/dataloaders/dataloader.py#L262
        Shape:
            label, (240, 144, 240)
            label_downscale, if downsample==4, then (60, 36, 60)
        """
        if downscale == 1:
            return label
        
        ds = downscale
        small_size = (
            self.grid_size[0] // ds,
            self.grid_size[1] // ds,
            self.grid_size[2] // ds,
        )  # small size
        label_downscale = np.zeros(small_size, dtype=np.uint8)
        empty_t = 0.95 * ds * ds * ds  # threshold
        s01 = small_size[0] * small_size[1]
        label_i = np.zeros((ds, ds, ds), dtype=np.int32)

        for i in range(small_size[0] * small_size[1] * small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])

            label_i[:, :, :] = label[
                x * ds : (x + 1) * ds, y * ds : (y + 1) * ds, z * ds : (z + 1) * ds
            ]
            label_bin = label_i.flatten()

            zero_count_0 = np.array(np.where(label_bin == 0)).size
            zero_count_255 = np.array(np.where(label_bin == 255)).size

            zero_count = zero_count_0 + zero_count_255
            if zero_count > empty_t:
                label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
            else:
                label_i_s = label_bin[
                    np.where(np.logical_and(label_bin > 0, label_bin < 255))
                ]
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
        
        return label_downscale
    
    def compute_CP_mega_matrix(self, target, is_binary=False):
        """
        Parameters
        ---------
        target: (H, W, D)
            contains voxels semantic labels

        is_binary: bool
            if True, return binary voxels relations else return 4-way relations
        """
        
        label = target.reshape(-1)
        label_row = label
        N = label.shape[0]
        
        # [32, 32, 4] ==> [16, 16, 2] ==> 512 super voxels
        super_voxel_size = [i // 2 for i in target.shape]
        
        if is_binary:
            matrix = np.zeros((2, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)
        else:
            # [num_way, num_voxel, num_super_voxel]
            matrix = np.zeros((4, N, super_voxel_size[0] * super_voxel_size[1] * super_voxel_size[2]), dtype=np.uint8)

        for xx in range(super_voxel_size[0]):
            for yy in range(super_voxel_size[1]):
                for zz in range(super_voxel_size[2]):
                    col_idx = xx * (super_voxel_size[1] * super_voxel_size[2]) + yy * super_voxel_size[2] + zz
                    label_col_megas = np.array([
                        target[xx * 2,     yy * 2,     zz * 2],
                        target[xx * 2 + 1, yy * 2,     zz * 2],
                        target[xx * 2,     yy * 2 + 1, zz * 2],
                        target[xx * 2,     yy * 2,     zz * 2 + 1],
                        target[xx * 2 + 1, yy * 2 + 1, zz * 2],
                        target[xx * 2 + 1, yy * 2,     zz * 2 + 1],
                        target[xx * 2,     yy * 2 + 1, zz * 2 + 1],
                        target[xx * 2 + 1, yy * 2 + 1, zz * 2 + 1],
                    ])
                    label_col_megas = label_col_megas[label_col_megas != 255]
                    for label_col_mega in label_col_megas:
                        label_col = np.ones(N) * label_col_mega
                        if not is_binary:
                            matrix[0, (label_row != 255) & (label_col == label_row) & (label_col != 0), col_idx] = 1.0 # non non same
                            matrix[1, (label_row != 255) & (label_col != label_row) & (label_col != 0) & (label_row != 0), col_idx] = 1.0 # non non diff
                            matrix[2, (label_row != 255) & (label_row == label_col) & (label_col == 0), col_idx] = 1.0 # empty empty
                            matrix[3, (label_row != 255) & (label_row != label_col) & ((label_row == 0) | (label_col == 0)), col_idx] = 1.0 # nonempty empty
                        else:
                            matrix[0, (label_row != 255) & (label_col != label_row), col_idx] = 1.0 # diff
                            matrix[1, (label_row != 255) & (label_col == label_row), col_idx] = 1.0 # same
        
        return matrix

    def init_voxel_coordinates(self):
        X, Y, Z = self.grid_size
        min_bound = self.point_cloud_range[:3] + self.voxel_size / 2
        
        xs = torch.arange(min_bound[0], self.point_cloud_range[3], self.voxel_size[0]).view(X, 1, 1).expand(X, Y, Z)
        ys = torch.arange(min_bound[1], self.point_cloud_range[4], self.voxel_size[1]).view(1, Y, 1).expand(X, Y, Z)
        zs = torch.arange(min_bound[2], self.point_cloud_range[5], self.voxel_size[2]).view(1, 1, Z).expand(X, Y, Z)
        
        # [X, Y, Z, 3]
        self.voxel_centers = torch.stack((xs, ys, zs), dim=-1)

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans, bda_mat):
        # project 3D point cloud (after bev-aug) onto multi-view images for corresponding 2D coordinates
        inv_bda = bda_mat.inverse()
        points = (inv_bda @ points.unsqueeze(-1)).squeeze(-1)
        
        # from lidar to camera
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = torch.cat((points, torch.ones((points.shape[0], 1, 1, 1))), dim=2)
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd

    def compute_local_frustums(self, projected_uvd, voxel_labels, img_size, frustum_size=8):
        H, W, D = voxel_labels.shape
        ranges = [(i * 1.0 / frustum_size, (i * 1.0 + 1) / frustum_size) for i in range(frustum_size)]
        local_frustum_masks = []
        local_frustum_class_dists = []
        pix_x, pix_y, pix_z = projected_uvd[:, 0], projected_uvd[:, 1], projected_uvd[:, 2]
        img_H, img_W = img_size
        
        for y in ranges:
            for x in ranges:
                start_x = x[0] * img_W
                end_x = x[1] * img_W
                start_y = y[0] * img_H
                end_y = y[1] * img_H
                
                # whether the projected voxels are inside the local patch
                local_frustum = (pix_x >= start_x) & (pix_x < end_x) & \
                        (pix_y >= start_y) & (pix_y < end_y) & (pix_z > 0)
                mask = (voxel_labels != 255) & local_frustum.reshape(H, W, D)
                
                classes, cnts = torch.unique(voxel_labels[mask], return_counts=True)
                class_counts = torch.zeros(self.n_classes)
                class_counts[classes.long()] = cnts.float()
                
                local_frustum_masks.append(mask)
                local_frustum_class_dists.append(class_counts)
        
        # (num_frustum * num_frustum, X, Y, Z)
        frustums_masks, frustums_class_dists = torch.stack(local_frustum_masks), torch.stack(local_frustum_class_dists)
        
        return frustums_masks, frustums_class_dists

    def __call__(self, results):
        voxel_labels = results['gt_occ']
        
        # downsample labels by 8x
        # downsampled_voxel_labels = self._downsample_label(voxel_labels, downscale=8)

        # create cp mega matrix
        # results['CP_mega_matrix'] = self.compute_CP_mega_matrix(downsampled_voxel_labels)
        
        # create frustum masks & frustum logits
        imgs, rots, trans, intrins, post_rots, post_trans, bda_mat = results['img_inputs'][:7]
        voxel_centers = self.voxel_centers.view(-1, 3)
        projected_uvd = self.project_points(voxel_centers, rots, trans, intrins, 
                                    post_rots, post_trans, bda_mat)[:, 0]
        
        img_h, img_w = imgs.shape[-2:]
        frustums_masks, frustums_class_dists = self.compute_local_frustums(projected_uvd, 
                            voxel_labels, img_size=(img_h, img_w))
        
        results['frustums_masks'] = frustums_masks
        results['frustums_class_dists'] = frustums_class_dists
        
        return results