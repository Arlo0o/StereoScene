
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
class CreateDepthFromOccupancy(object):
    def __init__(self, point_cloud_range, grid_size, downsample=False):
        self.grid_size = np.array(grid_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.grid_size
        self.downsample = downsample
        
        self.init_voxel_coordinates()
        
        self.class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]
    
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

    def _downsample_label(self, label, downscale=16):
        ds = downscale
        h, w = label.shape
        small_size = (
            h // ds,
            w // ds,
        )

        label_downscale = torch.zeros(small_size)
        empty_t = 0.95 * ds * ds  # threshold
        
        for i in range(small_size[0]):
            for j in range(small_size[1]):
                label_patch = label[i * ds : (i + 1) * ds, j * ds : (j + 1) * ds]
                label_patch = label_patch.flatten()
                
                zero_count_0 = (label_patch == 0).sum()
                zero_count_255 = (label_patch == 255).sum()
                zero_count = zero_count_0 + zero_count_255
                
                if zero_count > empty_t:
                    label_downscale[i, j] = 0 if zero_count_0 > zero_count_255 else 255
                else:
                    label_patch_valid = label_patch[(label_patch > 0) & (label_patch < 255)]
                    label_downscale[i, j] = torch.mode(label_patch_valid, dim=0)[0]
                    
        return label_downscale

    def __call__(self, results):
        target_occupancy = results['gt_occ']
        flatten_cls = target_occupancy.view(-1)
        flatten_xyz = self.voxel_centers.view(-1, 3)
        
        # select valid voxels
        unlabeled_mask = (flatten_cls == 0)
        ignored = (flatten_cls == 255)
        
  
        
        # project voxels onto the image plane
        imgs, rots, trans, intrins, post_rots, post_trans, bda_mat = results['img_inputs'][0][:7]
        projected_points = self.project_points(flatten_xyz, rots, trans, intrins, post_rots, post_trans, bda_mat)[:, 0]
        
        # filtering 
        img_h, img_w = imgs[0].shape[-2:]
        valid_mask = (projected_points[:, 0] >= 0) & \
                    (projected_points[:, 1] >= 0) & \
                    (projected_points[:, 0] <= img_w - 1) & \
                    (projected_points[:, 1] <= img_h - 1) & \
                    (projected_points[:, 2] > 0)
        
        '''
        create projected depth map
        '''
        img_depth = torch.zeros((img_h, img_w))
        depth_valid_mask = valid_mask & (~unlabeled_mask) & (~ignored)
        depth_projected_points = projected_points[depth_valid_mask]
        # sort and project
        depth_order = torch.argsort(depth_projected_points[:, 2], descending=True)
        depth_projected_points = depth_projected_points[depth_order]
        img_depth[depth_projected_points[:, 1].round().long(), depth_projected_points[:, 0].round().long()] = depth_projected_points[:, 2]
        
        '''
        create projected segmentation
        '''
        img_seg = torch.ones((img_h, img_w)) * 255
        seg_valid_mask = valid_mask
        flatten_cls = flatten_cls[seg_valid_mask]
        seg_projected_points = projected_points[seg_valid_mask]
        # sort and project
        seg_order = torch.argsort(seg_projected_points[:, 2], descending=True)
        seg_projected_points = seg_projected_points[seg_order]
        flatten_cls = flatten_cls[seg_order]
        img_seg[seg_projected_points[:, 1].round().long(), seg_projected_points[:, 0].round().long()] = flatten_cls
        
        # replace gt depth with occupancy-generated depth
        imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors = results['img_inputs'][0]
        list(results['img_inputs'])[0] = imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, img_depth.unsqueeze(0), sensor2sensors
        
        if self.downsample:
            img_seg = self._downsample_label(img_seg)
            
        results['img_seg'] = img_seg
        
        # self.visualize_image_labels(results['canvas'], img_depth, img_seg)
        
        return results
    
    def visualize_image_labels(self, img, img_depth, img_seg):
        out_path = 'debug_occupancy_projections'
        os.makedirs(out_path, exist_ok=True)
        
        import matplotlib.pyplot as plt
        
        # convert depth-map to depth-points
        depth_points = torch.nonzero(img_depth)
        depth_points = torch.stack((depth_points[:, 1], depth_points[:, 0], img_depth[depth_points[:, 0], depth_points[:, 1]]), dim=1)
        
        # overlay image with depth
        plt.figure(dpi=300)
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.scatter(depth_points[:, 0], depth_points[:, 1], s=2, c=depth_points[:, 2], alpha=0.7)
        plt.axis('off')
        plt.title('Image Depth')
        
        # overlay image with segmentation
        plt.subplot(2, 1, 2)
        img_color_seg = color_seg(img_seg).numpy().astype(np.uint8)
        alpha = 0.3
        img_color_seg = (1 - alpha) * img + alpha * img_color_seg
        plt.imshow(img_color_seg.astype(np.uint8))
        plt.axis('off')
        plt.title('Image Seg')
        
        plt.savefig(os.path.join(out_path, 'demo.png'))
        plt.close()
        
        pdb.set_trace()
    
    

@PIPELINES.register_module()
class CreateDepthFromLiDAR(object):
    def __init__(self, point_cloud_range, grid_size, projective_filter=True,
            label_mapping="semantickitti.yaml"):
        self.grid_size = np.array(grid_size)
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.voxel_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.grid_size
        
        self.class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]
        
        # how to filter the query lidar points
        self.projective_filter = projective_filter
        
        self.lidar_root = "/code/data/occupancy/semanticKITTI/lidar/velodyne/dataset/sequences/" # "./data/lidar/velodyne/dataset/sequences"
        self.lidarseg_root = "/code/data/occupancy/semanticKITTI/lidar/lidarseg/dataset/sequences/" #"./data/lidar/lidarseg/dataset/sequences"
          
        # mappings of training ids
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        
    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
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

    def __call__(self, results):
        ####################--------------------------1----------------------------#########################
        img_filename = results['img_filename'][0]
        seq_id, _, filename = img_filename.split("/")[-3:]
        
        # loading lidar points
        lidar_filename = os.path.join(self.lidar_root, seq_id, "velodyne", filename.replace(".png", ".bin"))
        lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
        lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
        
        # loading lidarseg labels
        lidarseg_filename = os.path.join(self.lidarseg_root, seq_id, "labels", filename.replace(".png", ".label"))
        lidarseg = np.fromfile(lidarseg_filename, dtype=np.uint32).reshape((-1, 1))
        lidarseg = lidarseg & 0xFFFF
        lidarseg = np.vectorize(self.learning_map.__getitem__)(lidarseg)
        # 0: ignored, 1 - 19 are valid labels
        lidarseg = torch.from_numpy(lidarseg.astype(np.int32)).float()
        flatten_seg = lidarseg.flatten()
        
        # project voxels onto the image plane
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][0][:6]
        projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)[:, 0]
        
        # create depth map
        img_h, img_w = imgs[0].shape[-2:]
        valid_mask = (projected_points[:, 0] >= 0) & \
                    (projected_points[:, 1] >= 0) & \
                    (projected_points[:, 0] <= img_w - 1) & \
                    (projected_points[:, 1] <= img_h - 1) & \
                    (projected_points[:, 2] > 0)
        
        # perform bird-eye-view augmentation for lidar_points
        bda_mat = results['img_inputs'][0][6]
        if bda_mat.shape[-1] == 4:
            homo_lidar_points = torch.cat((lidar_points, torch.ones(lidar_points.shape[0], 1)), dim=1)
            homo_lidar_points = homo_lidar_points @ bda_mat.t()
            lidar_points = homo_lidar_points[:, :3]
        else:
            lidar_points = lidar_points @ bda_mat.t()
        
        # perform range mask
        range_valid_mask = (lidar_points >= self.point_cloud_range[:3]) & (lidar_points <= self.point_cloud_range[3:])
        range_valid_mask = range_valid_mask.all(dim=1)

        
        lidarseg_mask = valid_mask
        lidarseg = torch.cat((lidar_points, lidarseg), dim=1)[lidarseg_mask]
        results['points_occ'] = lidarseg
        
        '''
        A simple validation, sampling the corresponding label in voxel for each point and check the consistency
        '''

        points_uvd = projected_points[lidarseg_mask]
        points_uvd[..., 0] /= img_w
        points_uvd[..., 1] /= img_h
        points_uvd[..., :2] = (points_uvd[..., :2] - 0.5) * 2
        results['points_uv'] = points_uvd.unsqueeze(dim=1)

        '''
        create projected depth map
        '''
        img_depth = torch.zeros((img_h, img_w))
        depth_projected_points = projected_points[valid_mask]
        # sort and project
        depth_order = torch.argsort(depth_projected_points[:, 2], descending=True)
        depth_projected_points = depth_projected_points[depth_order]
        img_depth[depth_projected_points[:, 1].round().long(), depth_projected_points[:, 0].round().long()] = depth_projected_points[:, 2]
        
        '''
        create image-view segmentation, label 0 is unlabeled,
        therefore we should only consider foreground classes (> 0)
        '''
        img_seg = torch.zeros((img_h, img_w))
        # using projective mask for filtering
        seg_valid_mask = valid_mask
        flatten_seg = flatten_seg[seg_valid_mask]
        seg_projected_points = projected_points[seg_valid_mask]
        # sort and project
        seg_order = torch.argsort(seg_projected_points[:, 2], descending=True)
        seg_projected_points = seg_projected_points[seg_order]
        flatten_seg = flatten_seg[seg_order]
        img_seg[seg_projected_points[:, 1].round().long(), seg_projected_points[:, 0].round().long()] = flatten_seg
        results['img_seg'] = img_seg
        
        # self.visualize(results['canvas'], img_depth, img_seg,out_path='debug_lidar_projections')   
        
        imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors, calib = results['img_inputs'][0]
        tmp1 = [imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, img_depth.unsqueeze(0), sensor2sensors, calib]

        ####################--------------------------2----------------------------#########################
        img_filename = results['img_filename'][1]
        seq_id, _, filename = img_filename.split("/")[-3:]
        
        # loading lidar points
        lidar_filename = os.path.join(self.lidar_root, seq_id, "velodyne", filename.replace(".png", ".bin"))
        lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
        lidar_points = torch.from_numpy(lidar_points[:, :3]).float()
        
        # loading lidarseg labels
        lidarseg_filename = os.path.join(self.lidarseg_root, seq_id, "labels", filename.replace(".png", ".label"))
        lidarseg = np.fromfile(lidarseg_filename, dtype=np.uint32).reshape((-1, 1))
        lidarseg = lidarseg & 0xFFFF
        lidarseg = np.vectorize(self.learning_map.__getitem__)(lidarseg)
        # 0: ignored, 1 - 19 are valid labels
        lidarseg = torch.from_numpy(lidarseg.astype(np.int32)).float()
        flatten_seg = lidarseg.flatten()
        
        # project voxels onto the image plane
        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][1][:6]
        projected_points = self.project_points(lidar_points, rots, trans, intrins, post_rots, post_trans)[:, 0]
        
        # create depth map
        img_h, img_w = imgs[0].shape[-2:]
        valid_mask = (projected_points[:, 0] >= 0) & \
                    (projected_points[:, 1] >= 0) & \
                    (projected_points[:, 0] <= img_w - 1) & \
                    (projected_points[:, 1] <= img_h - 1) & \
                    (projected_points[:, 2] > 0)
        
        # perform bird-eye-view augmentation for lidar_points
        bda_mat = results['img_inputs'][0][6]
        if bda_mat.shape[-1] == 4:
            homo_lidar_points = torch.cat((lidar_points, torch.ones(lidar_points.shape[0], 1)), dim=1)
            homo_lidar_points = homo_lidar_points @ bda_mat.t()
            lidar_points = homo_lidar_points[:, :3]
        else:
            lidar_points = lidar_points @ bda_mat.t()
        
        # perform range mask
        range_valid_mask = (lidar_points >= self.point_cloud_range[:3]) & (lidar_points <= self.point_cloud_range[3:])
        range_valid_mask = range_valid_mask.all(dim=1)

        
        lidarseg_mask = valid_mask
        lidarseg = torch.cat((lidar_points, lidarseg), dim=1)[lidarseg_mask]
        results['points_occ'] = lidarseg
        
        
        points_uvd = projected_points[lidarseg_mask]
        points_uvd[..., 0] /= img_w
        points_uvd[..., 1] /= img_h
        points_uvd[..., :2] = (points_uvd[..., :2] - 0.5) * 2
        results['points_uv'] = points_uvd.unsqueeze(dim=1)

        '''
        create projected depth map
        '''
        img_depth = torch.zeros((img_h, img_w))
        depth_projected_points = projected_points[valid_mask]
        # sort and project
        depth_order = torch.argsort(depth_projected_points[:, 2], descending=True)
        depth_projected_points = depth_projected_points[depth_order]
        img_depth[depth_projected_points[:, 1].round().long(), depth_projected_points[:, 0].round().long()] = depth_projected_points[:, 2]  
  

        '''
        create image-view segmentation, label 0 is unlabeled,
        therefore we should only consider foreground classes (> 0)
        '''
        img_seg = torch.zeros((img_h, img_w))
        # using projective mask for filtering
        seg_valid_mask = valid_mask
        flatten_seg = flatten_seg[seg_valid_mask]
        seg_projected_points = projected_points[seg_valid_mask]
        # sort and project
        seg_order = torch.argsort(seg_projected_points[:, 2], descending=True)
        seg_projected_points = seg_projected_points[seg_order]
        flatten_seg = flatten_seg[seg_order]
        img_seg[seg_projected_points[:, 1].round().long(), seg_projected_points[:, 0].round().long()] = flatten_seg
        results['img_seg'] = img_seg   
        
        imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, gt_depths, sensor2sensors, calib = results['img_inputs'][1]
        
        tmp2 = [imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, img_depth.unsqueeze(0), sensor2sensors, calib]
        results['img_inputs'] = [tmp1,tmp2]
        
 
               
        return results
        

    def visualize(self, img, img_depth, img_seg, out_path='debug_lidar_projections'):
        
        os.makedirs(out_path, exist_ok=True)
        
        import matplotlib.pyplot as plt
        
        # convert depth-map to depth-points
        depth_points = torch.nonzero(img_depth)
        depth_points = torch.stack((depth_points[:, 1], depth_points[:, 0], img_depth[depth_points[:, 0], depth_points[:, 1]]), dim=1)
        
        # overlay image with depth
        plt.figure(dpi=300)
        plt.imshow(img)
        plt.scatter(depth_points[:, 0], depth_points[:, 1], s=1, c=depth_points[:, 2], alpha=0.5)
        plt.axis('off')
        plt.title('Image Depth')
        
        plt.savefig(os.path.join(out_path, 'demo_depth.png'))
        plt.close()
        
        # overlay image with seg
        alpha = 0.5
        img_color_seg = color_seg(img_seg).numpy().astype(np.uint8)
        img_seg_mask = (img_seg > 0)
        blend_img_seg = img.copy()
        blend_img_seg[img_seg_mask] = alpha * blend_img_seg[img_seg_mask] + (1 - alpha) * img_color_seg[img_seg_mask]
        
        plt.figure(dpi=300)
        plt.imshow(blend_img_seg)
        plt.axis('off')
        plt.title('Image Seg')
        
        plt.savefig(os.path.join(out_path, 'demo_seg.png'))
        plt.close()
        
        # show imgseg == 0
        imgseg_zero_points = torch.nonzero(img_seg == 0)
        # overlay image with depth
        plt.figure(dpi=300)
        plt.imshow(img)
        plt.scatter(imgseg_zero_points[:, 1], imgseg_zero_points[:, 0], s=1, c='r', alpha=0.5)
        plt.axis('off')
        plt.title('Image Seg 0')
        
        plt.savefig(os.path.join(out_path, 'demo_seg_0.png'))
        plt.close()
        
        # show imgseg == 255
        imgseg_ignore_points = torch.nonzero(img_seg == 255)
        # overlay image with depth
        plt.figure(dpi=300)
        plt.imshow(img)
        plt.scatter(imgseg_ignore_points[:, 1], imgseg_ignore_points[:, 0], s=1, c='r', alpha=0.5)
        plt.axis('off')
        plt.title('Image Seg 255')
        
        plt.savefig(os.path.join(out_path, 'demo_seg_255.png'))
        plt.close()
        
        # pdb.set_trace()


def color_seg(seg):
    colors = [
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255],
    ]
    
    # convert long-type seg labels to uint8 color seg images
    output = torch.zeros(*seg.shape, 3).float()
    for cls_id, color in enumerate(colors):
        cls_mask = (seg == (cls_id + 1))
        output[cls_mask] = torch.tensor(color[:3]).float()
    
    return output
        