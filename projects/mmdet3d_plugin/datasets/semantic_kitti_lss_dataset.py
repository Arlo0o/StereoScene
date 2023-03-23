import copy
import tqdm
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.custom_3d import Custom3DDataset

import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.datasets.pipelines.loading import LoadOccupancy
from mmcv.parallel import DataContainer as DC
import random
import pdb, os
import glob
import numpy as np
from .semantic_kitti_dataset import CustomSemanticKITTIDataset

@DATASETS.register_module()
class CustomSemanticKITTILssDataset(CustomSemanticKITTIDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, random_camera=False, cbgs=False, repeat=1, load_multi_voxel=False, *args, **kwargs):
        super(CustomSemanticKITTILssDataset, self).__init__(*args, **kwargs)
        
        self.random_camera = random_camera
        self.all_camera_ids = list(self.camera_map.values())
        self.load_multi_voxel = load_multi_voxel
        self.multi_scales = ["1_1", "1_2", "1_4", "1_8", "1_16"]
        self.repeat = repeat
        self.cbgs = cbgs
        
        if self.repeat > 1:
            self.data_infos = self.data_infos * self.repeat
            random.shuffle(self.data_infos)
        
        # init class-balanced sampling
        self.data_infos = self.init_cbgs()
        
        self._set_group_flag()
        
    def prepare_cat_infos(self):
        tmp_file = 'semkitti_train_class_counts.npy'
        
        if not os.path.exists(tmp_file):
            class_counts_list = []
            for index in tqdm.trange(len(self)):
                info = self.data_infos[index]['voxel_path']
                assert info is not None    
                target_occ = np.load(info)
                
                # compute the class counts
                cls_ids, cls_counts = np.unique(target_occ, return_counts=True)
                class_counts = np.zeros(self.n_classes)
                
                cls_ids = cls_ids.astype(np.int)
                for cls_id, cls_count in zip(cls_ids, cls_counts):
                    # ignored
                    if cls_id == 255:
                        continue
                    
                    class_counts[cls_id] += cls_count
                
                class_counts_list.append(class_counts)
            
            # num_sample, num_class
            self.class_counts_list = np.stack(class_counts_list, axis=0)
            np.save(tmp_file, self.class_counts_list)
        else:
            self.class_counts_list = np.load(tmp_file)
    
    def init_cbgs(self):
        if not self.cbgs:
            return self.data_infos
        
        self.prepare_cat_infos()
        # remove unlabel class
        self.class_counts_list = self.class_counts_list[:, 1:]
        num_class = self.class_counts_list.shape[1]
        
        class_sum_counts = np.sum(self.class_counts_list, axis=0)
        sample_sum = class_sum_counts.sum()
        class_distribution = class_sum_counts / sample_sum
        
        # compute the balanced ratios
        frac = 1.0 / num_class
        ratios = frac / class_distribution
        ratios = np.log(1 + ratios)
        
        sampled_idxs_list = []
        for cls_id in range(num_class):
            # number of total points for this class
            num_class_sample_pts = class_sum_counts[cls_id] * ratios[cls_id]
            
            # get corresponding samples
            class_sample_valid_mask = (self.class_counts_list[:, cls_id] > 0)
            class_sample_valid_indices = class_sample_valid_mask.nonzero()[0]
            
            class_sample_points = self.class_counts_list[class_sample_valid_mask, cls_id]
            class_sample_prob = class_sample_points / class_sample_points.sum()
            class_sample_expectation = (class_sample_prob * class_sample_points).sum()
            
            # class_sample_mean = class_sample_points.mean()
            num_samples = int(num_class_sample_pts / class_sample_expectation)
            sampled_idxs = np.random.choice(class_sample_valid_indices, size=num_samples, p=class_sample_prob)
            sampled_idxs_list.extend(sampled_idxs)
        
        sampled_infos = [self.data_infos[i] for i in sampled_idxs_list]
        
        return sampled_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        return example

    def get_ann_info(self, index):
        info = self.data_infos[index]['voxel_path']
        if info is None:
            return None

        if self.load_multi_voxel:
            annos = []
            for scale in self.multi_scales:
                scale_info = info.replace('1_1', scale)
                annos.append(np.load(scale_info))
            
            return annos
        else:
            return np.load(info)

    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        '''
        
        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
        )
        
        # load images, intrins, extrins, voxels
        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []
        for cam_type in self.camera_used:
            if self.random_camera:
                cam_type = random.choice(self.all_camera_ids)

            image_paths.append(info['img_{}_path'.format(int(cam_type))])
      
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
      
            cam_intrinsics.append(info['P{}'.format(int(cam_type))])
       
            lidar2cam_rts.append(info['T_velo_2_cam'])


        calib_info = self.read_calib_file(info['calib_path'])
        calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * self.dynamic_baseline(calib_info)  
        # calib = np.reshape(calib_info['P2'], [3, 4])[0, 0] * 0.54

        input_dict.update(
            dict(
                img_filename=image_paths,    ###### image2, imag3
       
                lidar2img=lidar2img_rts,
          
                cam_intrinsic=cam_intrinsics,
            
                lidar2cam=lidar2cam_rts,

                calib = calib
            ))
    
        # ground-truth in shape (256, 256, 32), XYZ order
        # TODO: how to do bird-eye-view augmentation for this? 
        input_dict['gt_occ'] = self.get_ann_info(index)
        return input_dict

    def read_calib_file(self, filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    def dynamic_baseline(self, calib_info):
        P3 =np.reshape(calib_info['P3'], [3,4])
        P =np.reshape(calib_info['P2'], [3,4])
        baseline = P3[0,3]/(-P3[0,0]) - P[0,3]/(-P[0,0])
        return baseline

    def evaluate(self, results, logger=None, **kwargs):
        if 'ssc_scores' in results:
            ssc_scores = results['ssc_scores']
            
            class_ssc_iou = ssc_scores['iou_ssc'].tolist()
            res_dic = {
                "SC_Precision": ssc_scores['precision'].item(),
                "SC_Recall": ssc_scores['recall'].item(),
                "SC_IoU": ssc_scores['iou'],
                "SSC_mIoU": ssc_scores['iou_ssc_mean'],
            }
        else:
            assert 'ssc_results' in results
            ssc_results = results['ssc_results']
            completion_tp = sum([x[0] for x in ssc_results])
            completion_fp = sum([x[1] for x in ssc_results])
            completion_fn = sum([x[2] for x in ssc_results])
            
            tps = sum([x[3] for x in ssc_results])
            fps = sum([x[4] for x in ssc_results])
            fns = sum([x[5] for x in ssc_results])
            
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / \
                    (completion_tp + completion_fp + completion_fn)
            iou_ssc = tps / (tps + fps + fns + 1e-5)
            
            class_ssc_iou = iou_ssc.tolist()
            res_dic = {
                "SC_Precision": precision,
                "SC_Recall": recall,
                "SC_IoU": iou,
                "SSC_mIoU": iou_ssc[1:].mean(),
            }
        
        class_names = [
            'unlabeled', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
            'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
            'pole', 'traffic-sign'
        ]
        for name, iou in zip(class_names, class_ssc_iou):
            res_dic["SSC_{}_IoU".format(name)] = iou
        
        eval_results = {}
        for key, val in res_dic.items():
            eval_results['semkitti_{}'.format(key)] = round(val * 100, 2)
        
        # add two main metrics to serve as the sort metric
        eval_results['semkitti_combined_IoU'] = eval_results['semkitti_SC_IoU'] + eval_results['semkitti_SSC_mIoU']
        
        if logger is not None:
            logger.info('SemanticKITTI SSC Evaluation')
            logger.info(eval_results)
        
        return eval_results
        
        