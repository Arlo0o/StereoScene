import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
import mmcv
import collections 

from mmdet.models import DETECTORS
from mmdet3d.models import builder, losses
from collections import OrderedDict
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner.checkpoint import _load_checkpoint, load_state_dict
from sklearn.metrics import confusion_matrix as CM
from .bevdepth import BEVDepth, BEVDepth4D
from projects.mmdet3d_plugin.utils import fast_hist_crop
from projects.mmdet3d_plugin.models.utils import GridMask

import numpy as np
import time
import pdb

@DETECTORS.register_module()
class BEVDepthOccupancy(BEVDepth):
    def __init__(self, 
            loss_cfg=None,
            use_grid_mask=False,
            disable_loss_depth=False,
            **kwargs):
        super().__init__(**kwargs)
        
        self.loss_cfg = loss_cfg
        self.use_grid_mask = use_grid_mask
        self.disable_loss_depth = disable_loss_depth
        
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        
        self.record_time = False
        self.time_stats = collections.defaultdict(list)
    
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape  ## [2, 3, 384, 1280]
        imgs = imgs.view(B * N, C, imH, imW)
        
        if self.use_grid_mask:
            imgs = self.grid_mask(imgs)
        
        x = self.img_backbone(imgs)  ## 5 [2, 48, 96, 320]  [2, 80, 48, 160]  [2, 224, 24, 80] [2, 640, 12, 40]  [2, 2560, 12, 40]

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    @force_fp32()
    def bev_encoder(self, x):  ## [4, 128, 128, 128, 16]
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        x = self.img_bev_encoder_backbone(x) ### [4, 128, 128, 128, 16] [4, 256, 64, 64, 8] [4, 512, 32, 32, 4]
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['bev_encoder'].append(t1 - t0)
        
        x = self.img_bev_encoder_neck(x)  ### [4, 384, 128, 128, 16]
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['bev_neck'].append(t2 - t1)
        
        return x
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()

 
        img, img2 = img[0], img[1]
        B, N, C, H, W = img[0].shape
       
        feature_out  = self.image_encoder(  torch.cat([ img[0], img2[0] ],0)   ) ### [B N C H W] [2, 1, 640, 24, 80]
       
        x= feature_out[:B ]
        x2= feature_out[B: ]

        img_feats = x.clone()
         
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['img_encoder'].append(t1 - t0)

        # img: imgs, rots, trans, intrins, post_rots, post_trans, gt_depths, sensor2sensors
        rots, trans, intrins, post_rots, post_trans, bda = img[1:7]  ####    bda[2,3,3]
        rots2, trans2, intrins2, post_rots2, post_trans2, bda2 = img2[1:7]  ####    bda[2,3,3]
        
        mlp_input = self.img_view_transformer.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)  ###  [2, 1, 30]
        mlp_input2 = self.img_view_transformer.get_mlp_input(rots2, trans2, intrins2, post_rots2, post_trans2, bda2)  ###  [2, 1, 30]
        
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]  ### len=7
        geo_inputs2 = [rots2, trans2, intrins2, post_rots2, post_trans2, bda2, mlp_input2]  ### len=7 

        calib = img[-1]
        x, depth = self.img_view_transformer([x] + geo_inputs + [x2] + geo_inputs2 + [calib]+ [img, img2] )  #### x[2, 128, 128, 128, 16]  depth[B N H W] [2, 112, 24, 80]


        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['view_transformer'].append(t2 - t1)
        
        x = self.bev_encoder(x)
        if type(x) is not list:
            x = [x]
        
        return x, depth, img_feats

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        
        voxel_feats, depth, img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (voxel_feats, img_feats, depth )
    
    @force_fp32(apply_to=('pts_feats'))
    def forward_pts_train(
            self,
            pts_feats,
            gt_occ=None,
            points_occ=None,
            img_metas=None,
            img_feats=None,
            points_uv=None,
            **kwargs,
        ):
        
        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        
        outs = self.pts_bbox_head(
            voxel_feats=pts_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=points_uv,
            **kwargs,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_head'].append(t1 - t0)
        
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            target_voxels=gt_occ,
            output_points=outs['output_points'],
            target_points=points_occ,
            img_metas=img_metas,
            **kwargs,
        )
        
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['loss_occ'].append(t2 - t1)
        
        return losses
    
    def forward_train(self,
            points=None,
            img_metas=None,
            img_inputs=None,
            gt_occ=None,
            points_occ=None,
            points_uv=None,
            **kwargs,
        ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        
        # extract bird-eye-view features from perspective images
        voxel_feats, img_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        
        # training losses
        losses = dict()
        
        if self.record_time:        
            torch.cuda.synchronize()
            t0 = time.time()
        
        if not self.disable_loss_depth:
            losses['loss_depth'] = self.img_view_transformer.get_depth_loss(img_inputs[0][7], depth)  ### ce loss


        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['loss_depth'].append(t1 - t0)
            
        if self.img_bev_encoder_backbone.crp3d:
            losses['loss_rel_ce'] = self.img_bev_encoder_backbone.crp_loss(
                CP_mega_matrices=kwargs['CP_mega_matrix'],
            )
        
        if self.img_view_transformer.imgseg:
            losses['loss_imgseg'] = self.img_view_transformer.get_seg_loss(
                seg_labels=kwargs['img_seg'],
            )
        
        losses_occupancy = self.forward_pts_train(voxel_feats, gt_occ, 
                        points_occ, img_metas, img_feats=img_feats, points_uv=points_uv, **kwargs)
        losses.update(losses_occupancy)
        
        def logging_latencies():
            # logging latencies
            avg_time = {key: sum(val) / len(val) for key, val in self.time_stats.items()}
            sum_time = sum(list(avg_time.values()))
            out_res = ''
            for key, val in avg_time.items():
                out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)
            
            print(out_res)
        
        if self.record_time:
            logging_latencies()
        
        return losses
        
    def forward_test(self,
            img_metas=None,
            img_inputs=None,
            **kwargs,
        ):
        
        return self.simple_test(img_metas, img_inputs, **kwargs)
    
    def simple_test(self, img_metas, img=None, rescale=False, points_occ=None, gt_occ=None, points_uv=None):
        
        voxel_feats, img_feats, depth = self.extract_feat(points=None, img=img, img_metas=img_metas)        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            points_uv=points_uv,
        )
        
        # evaluate nusc lidar-seg
        if output['output_points'] is not None and points_occ is not None:
            output['evaluation_semantic'] = self.simple_evaluation_semantic(output['output_points'], points_occ, img_metas)
        else:
            output['evaluation_semantic'] = 0
            
        # evaluate voxel 
        output['output_voxels'] = F.interpolate(output['output_voxels'][0], 
                    size=gt_occ.shape[1:], mode='trilinear', align_corners=False)
        output['target_voxels'] = gt_occ
        
        return output
    
    def post_process_semantic(self, pred_occ):
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        score, color = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        
        return color

    def simple_evaluation_semantic(self, pred, gt, img_metas):
        pred = torch.argmax(pred[0], dim=1).cpu().numpy()
        gt = gt[0].cpu().numpy()
        gt = gt[:, 3].astype(np.int)
        unique_label = np.arange(16)
        
        hist = fast_hist_crop(pred, gt, unique_label)
        
        return hist
    
    def evaluation_semantic(self, pred, gt, img_metas):
        import open3d as o3d

        assert pred.shape[0] == 1
        pred = pred[0]
        gt_ = gt[0].cpu().numpy()
        
        x = np.linspace(0, pred.shape[0] - 1, pred.shape[0])
        y = np.linspace(0, pred.shape[1] - 1, pred.shape[1])
        z = np.linspace(0, pred.shape[2] - 1, pred.shape[2])
    
        X, Y, Z = np.meshgrid(x, y, z,  indexing='ij')
        vv = np.stack([X, Y, Z], axis=-1)
        pred_fore_mask = pred > 0
        
        if pred_fore_mask.sum() == 0:
            return None
        
        # select foreground 3d voxel vertex
        vv = vv[pred_fore_mask]
        vv[:, 0] = (vv[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
        vv[:, 1] = (vv[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
        vv[:, 2] = (vv[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vv)
        
        # for every lidar point, search its nearest *foreground* voxel vertex as the semantic prediction
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        indices = []
        
        for vert in gt_[:, :3]:
            _, inds, _ = kdtree.search_knn_vector_3d(vert, 1)
            indices.append(inds[0])
        
        gt_valid = gt_[:, 3].astype(np.int)
        pred_valid = pred[pred_fore_mask][np.array(indices)]
        
        mask = gt_valid > 0
        cm = CM(gt_valid[mask] - 1, pred_valid[mask] - 1, labels=np.arange(16))
        cm = cm.astype(np.float32)
        
        return cm
        

@DETECTORS.register_module()
class BEVDepthOccupancy4D(BEVDepthOccupancy):
    def prepare_voxel_feat(self, img, rot, tran, intrin, 
                post_rot, post_tran, bda, mlp_input):
        
        x = self.image_encoder(img)
        img_feats = x.clone()
        
        voxel_feat, depth = self.img_view_transformer([x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        
        return voxel_feat, depth, img_feats

    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N//2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, bda = inputs[1:7]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        voxel_feat_list = []
        img_feat_list = []
        depth_list = []
        key_frame = True # back propagation for key frame only
        
        for img, rot, tran, intrin, post_rot, \
            post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
                
            mlp_input = self.img_view_transformer.get_mlp_input(
                rots[0], trans[0], intrin,post_rot, post_tran, bda)
            inputs_curr = (img, rot, tran, intrin, post_rot, post_tran, bda, mlp_input)
            if not key_frame:
                with torch.no_grad():
                    voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
            else:
                voxel_feat, depth, img_feats = self.prepare_voxel_feat(*inputs_curr)
            
            voxel_feat_list.append(voxel_feat)
            img_feat_list.append(img_feats)
            depth_list.append(depth)
            key_frame = False
        
        voxel_feat = torch.cat(voxel_feat_list, dim=1)
        x = self.bev_encoder(voxel_feat)
        if type(x) is not list:
            x = [x]

        return x, depth_list[0], img_feat_list[0]
        