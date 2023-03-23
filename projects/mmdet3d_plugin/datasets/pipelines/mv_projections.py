import mmcv
import numpy as np
import torch
import os

from mmdet.datasets.builder import PIPELINES

import matplotlib.pyplot as plt
import pdb

# create voxel-level labels from lidarseg points
@PIPELINES.register_module()
class MultiViewProjections(object):
    def __init__(self,
            create_voxel_projections=False,
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
            voxel_size=[0.4, 0.4, 0.5], 
            grid_size=[256, 256, 16],
        ):
        # create point projections & (optional) voxel projections
        self.create_voxel_projections = create_voxel_projections
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = np.array(voxel_size)
        self.grid_size = np.array(grid_size)
        
        self.init_voxel_coordinates()
    
    def debug(self, projections, img_canvas, labels=None):
        out_path = 'debug_mv_projections'
        os.makedirs(out_path, exist_ok=True)
        
        # 大概一半的 lidar points 都没有对应的图像像素，可能是由于 resize 和 crop 导致的
        valid_sum = 0
        for cam_index, img in enumerate(img_canvas):
            cam_projections = projections[:, cam_index]
            h, w = img.shape[:2]
            # valid filter
            is_valid = (cam_projections[:, 0] >= 0) & \
                    (cam_projections[:, 1] >= 0) & \
                    (cam_projections[:, 0] <= w - 1) & \
                    (cam_projections[:, 1] <= h - 1) & \
                    (cam_projections[:, 2] > 0)
            
            print('cam {}, valid points = {} / {}'.format(cam_index, is_valid.sum(), is_valid.shape[0]))
            cam_projections = cam_projections[is_valid]
            valid_sum += is_valid.sum()
            
            plt.figure()
            plt.imshow(img)
            plt.scatter(cam_projections[:, 0], cam_projections[:, 1], s=5, c=cam_projections[:, 2], alpha=0.7)
            plt.axis('off')
            plt.savefig(os.path.join(out_path, 'cam{}_proj.png'.format(cam_index)))
            plt.close()
        
        print('input image', valid_sum, projections.shape[0])
        
        pdb.set_trace()
    
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
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
    def __call__(self, results):
        _, rots, trans, intrins, post_rots, post_trans, bda_rot = results['img_inputs'][:7]
        bda_mat = results['bda_mat']
        
        points = results['points_occ'][:, :3]
        points = torch.from_numpy(points).float()
        
        if self.create_voxel_projections:
            voxel_centers = self.voxel_centers.view(-1, 3)
            num_voxel_points = voxel_centers.shape[0]
            concat_points = torch.cat((voxel_centers, points))
            
            concat_points_uvd = self.project_points(concat_points, rots, trans, intrins, 
                        post_rots, post_trans, bda_mat)
            
            # self.debug(concat_points_uvd, results['canvas'])
            
            img_h, img_w = results['img_inputs'][0].shape[-2:]
            concat_points_uvd[..., 0] /= img_w
            concat_points_uvd[..., 1] /= img_h
            concat_points_uvd[..., :2] = (concat_points_uvd[..., :2] - 0.5) * 2
            
            results['points_uv'] = concat_points_uvd
        else:
            points_uvd = self.project_points(points, rots, trans, intrins, 
                        post_rots, post_trans, bda_mat)
            
            # self.debug(points_uvd, results['canvas'])
            
            img_h, img_w = results['img_inputs'][0].shape[-2:]
            points_uvd[..., 0] /= img_w
            points_uvd[..., 1] /= img_h
            points_uvd[..., :2] = (points_uvd[..., :2] - 0.5) * 2
            results['points_uv'] = points_uvd
        
        return results