
import trimesh
import mmcv
import numpy as np

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
import yaml, os
import torch

import pdb

@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=True, use_semantic=False):
        self.to_float32 = to_float32
        self.use_semantic = use_semantic

    
    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        occ_size_ori = [600, 600, 48]

   
        
        root_path = './point_mesh_voxel/'

        
        root_path = './semantic_label/'
        rel_path = 'scene_{0}/dense_voxels_with_semantic/{1}.npy'.format(results['scene_token'], results['lidar_token'])
        # [z, x, y, cls]
        pcd = np.load(root_path + rel_path)
        pcd_np = pcd[..., [2,1,0]].astype(np.int)
        occ_np = np.zeros(occ_size_ori).astype(np.int)
        semantics = pcd[..., -1]
        semantics[semantics == 0] = 255
        
        # restrict boundaries, why not just slice ?
        for i in range(3):
            pcd_np[:, i][pcd_np[:, i] >= occ_size_ori[i]] = occ_size_ori[i] - 1    
        
        occ_np[pcd_np[:, 0], pcd_np[:, 1], pcd_np[:, 2]] = semantics
        
        # [0, 600], [0, 600], [0, 48], voxel size = [0.2, 0.2, 1/6?]
        occ_np_cropped = occ_np[300 - results['occ_size'][0] // 2: 300 + results['occ_size'][0] // 2, \
                                300 - results['occ_size'][1] // 2: 300 + results['occ_size'][1] // 2, \
                                24 - results['occ_size'][2] // 2: 24 + results['occ_size'][2] // 2]
        results['gt_occ'] = occ_np_cropped
        '''
        
        results['gt_occ'] = np.ones((600, 600, 48)).astype(np.float32)
        '''
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

@PIPELINES.register_module()
class LoadMesh(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=True, load_semantic=False):
        self.to_float32 = to_float32
        self.load_semantic = load_semantic

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """


        occ_size_ori = [600, 600, 48]
        root_path = './semantic_label/'
        rel_path = 'scene_{0}/vertice/{1}.npy'.format(results['scene_token'], results['lidar_token'])
        pcd = np.load(root_path + rel_path)

        gt_mask = (pcd[:, 0] > results['pc_range'][0]) * \
              (pcd[:, 0] < results['pc_range'][3]) * \
              (pcd[:, 1] > results['pc_range'][1]) * \
              (pcd[:, 1] < results['pc_range'][4]) * \
              (pcd[:, 2] > results['pc_range'][2]) * \
              (pcd[:, 2] < results['pc_range'][5])
        pcd = pcd[gt_mask]
        results['points_occ'] = pcd




        if self.load_semantic:

            import yaml, os
            label_mapping = './BEVFormer/util/nuscenes.yaml'
            with open(label_mapping, 'r') as stream:
                nuscenesyaml = yaml.safe_load(stream)
                learning_map = nuscenesyaml['learning_map']

            lidarseg_labels_filename = os.path.join('./BEVFormer/nus_lidarseg/lidarseg/v1.0-trainval',
                                                        results['lidarseg'])

            points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            points_label = np.vectorize(learning_map.__getitem__)(points_label)

            pc0 = np.fromfile(results['pts_filename'],
                              dtype=np.float32,
                              count=-1).reshape(-1, 5)[..., :3]
            pcd_np = np.concatenate([pc0, points_label], axis=-1)
            mask = (pcd_np[:, 0] > results['pc_range'][0]) * \
               (pcd_np[:, 1] > results['pc_range'][1]) * \
               (pcd_np[:, 2] > results['pc_range'][2]) * \
               (pcd_np[:, 0] < results['pc_range'][3]) * \
               (pcd_np[:, 1] < results['pc_range'][4]) * \
               (pcd_np[:, 2] < results['pc_range'][5]) 

            pcd_np = pcd_np[mask]
            results['gt_semantic'] = pcd_np

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str


@PIPELINES.register_module()
class LoadSemanticPoint(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=True, cls_metas='nuscenes.yaml', lidar_seg_path=None, filter_points=False):
        self.to_float32 = to_float32
        
        self.cls_metas = cls_metas
        with open(cls_metas, 'r') as stream:
            nusc_cls_metas = yaml.safe_load(stream)
            self.learning_map = nusc_cls_metas['learning_map']
        
        fsd_path = './nuScenes/lidarseg/v1.0-trainval'
        a100_path = './nuscenes/origin/lidarseg/v1.0-trainval'
        self.on_a100 = False
        
        if os.path.exists(a100_path):
            self.on_a100 = True
            self.lidar_seg_path = a100_path
        else:
            self.lidar_seg_path = fsd_path
        
        self.filter_points = filter_points

    def __call__(self, results):
        lidarseg_labels_filename = os.path.join(self.lidar_seg_path, results['lidarseg'])
        points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
        points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
        
        if self.on_a100:
            pts_filename = results['pts_filename'].replace('./data/nuscenes/', './det3d/nuscenes/origin/')
        else:
            pts_filename = results['pts_filename']
        pc0 = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(-1, 5)[..., :3]
        
        # perform bird-eye-view augmentation for 3D points
        pc0 = (results['bda_mat'] @ torch.from_numpy(pc0).unsqueeze(-1)).squeeze(-1).float().numpy()
        pcd_np = np.concatenate([pc0, points_label], axis=-1)
        

        
        if self.filter_points:
            mask = (pcd_np[:, 0] > results['pc_range'][0]) & \
                (pcd_np[:, 1] > results['pc_range'][1]) & \
                (pcd_np[:, 2] > results['pc_range'][2]) & \
                (pcd_np[:, 0] < results['pc_range'][3]) & \
                (pcd_np[:, 1] < results['pc_range'][4]) & \
                (pcd_np[:, 2] < results['pc_range'][5]) 
            pcd_np = pcd_np[mask]
        
        results['points_occ'] = pcd_np

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str