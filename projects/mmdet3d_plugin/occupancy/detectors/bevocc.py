# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------



from tkinter.messagebox import NO
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
import mmdet3d
import mcubes
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import pdb
from sklearn.metrics import confusion_matrix as CM
import time

@DETECTORS.register_module()
class BEVOcc(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False
                 ):

        super(BEVOcc,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }
        self.cm = np.zeros((16, 16)).astype(np.float32)
        self.cd = 0
        self.count = 0
        self.lidar_tokens = []


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_occ,
                          img_metas,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_occ, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img_metas=None,
                      gt_occ=None,
                      img=None
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
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        if prev_img.shape[1] > 0:
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
        else:
            prev_bev = None

        img_metas = [each[len_queue-1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        
        # feature maps with strides (8, 16, 32)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_occ, img_metas, prev_bev)
        losses.update(losses_pts)
        
        return losses

    def forward_test(self, img_metas, img=None, points_occ=None, gt_occ=None, gt_semantic=None, **kwargs):
        #self.train()
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        
        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        start = time.time()
        new_prev_bev, output = self.simple_test(
            img_metas, img, prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        
        # gt_semantic: N x 4 array in [x, y, z, cls]
        if gt_semantic is None:
            pred_occ, pred_ground = self.post_process(output)
            if pred_ground is None:
                eval_results = self.evaluation(pred_occ.cpu().numpy(), points_occ, img_metas[0])
            else:
                eval_results = self.evaluation(pred_occ.cpu().numpy(), points_occ, img_metas[0], pred_ground.cpu().numpy())
            if not np.isnan(eval_results.sum()):
                self.cd += eval_results
                self.count += 1
            print(self.cd / self.count, self.count)

        if gt_semantic is not None:
            pred_semantic = self.post_process_semantic(output)
            eval_semantic = self.evaluation_semantic(pred_semantic.cpu().numpy(), gt_semantic, img_metas[0])
            
            # self.cm += eval_semantic
                        
            # confusion matrix to mean IoU
            # mean_ious = cm_to_ious(self.cm)
            
            # format and logging
            # print(format_results(mean_ious))
            
            return {'evaluation_semantic': eval_semantic}
        else:
            return {'evaluation': eval_results}
        
    def post_process(self, output):
        pred_occ = output['occ_preds']
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]
        
        # 0 for background, 1 for foreground occupancy
        if pred_occ.shape[1] > 1:
            score, color = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
            pred_occ = (color > 0).float()
        else:
            pred_occ = torch.sigmoid(pred_occ[:, 0])
        
        if 'ground_preds' in output.keys():
            pred_ground = output['ground_preds']
            
            if type(pred_ground) == list:
                pred_ground = pred_ground[-1]

            if pred_ground.shape[1] > 1:
                _, color_ground = torch.max(torch.softmax(pred_ground, dim=1), dim=1)
                pred_ground = (color_ground > 0).float()
            else:
                pred_ground = torch.sigmoid(pred_ground[:, 0])
            
            #pred_occ[pred_ground > 0.5] = pred_ground[pred_ground > 0.5]

            return pred_occ, pred_ground
        else:
            return pred_occ, None

    def post_process_semantic(self, output):
        pred_occ = output['occ_preds']
        if type(pred_occ) == list:
            pred_occ = pred_occ[-1]

        score, color = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
        
        if 'ground_preds' in output.keys():
        
            pred_ground = output['ground_preds']  
            if type(pred_ground) == list:
                pred_ground = pred_ground[-1]
    
            _, color_ground = torch.max(torch.softmax(pred_ground, dim=1), dim=1)
                
            color_ground[color_ground > 0] = color_ground[color_ground > 0] + 10
            
            color[color == 11] = 15
            color[color == 12] = 16 
    
            color_ground[color_ground == 0] = color[color_ground == 0]
    
            return color_ground
        else:
            return color

    def evaluation(self, pred_occ, points_occ, img_metas, pred_ground=None):
        import open3d as o3d
        import os
        occ_results = []
        assert pred_occ.shape[0] == 1
        for i in range(pred_occ.shape[0]):
            # occ_hat_padded = np.pad(
            #         pred_occ[i], 1, 'constant', constant_values=-1e6)
            # vertices, triangles = mcubes.marching_cubes(occ_hat_padded, 0.5)
            # vertices -= 0.5
            # vertices -= 1
            
   
            vertices, triangles = mcubes.marching_cubes(pred_occ[i], 0.5)
            vertices_ori = vertices.copy()
            
            # recover xyz coordinates
            vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
            vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
            vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]
            vertices_object = vertices.copy()
            
            # occ_hat_padded = np.pad(
            #         pred_ground[i], 1, 'constant', constant_values=-1e6)
            # vertices, triangles = mcubes.marching_cubes(occ_hat_padded, 0.5)
            # vertices -= 0.5
            # vertices -= 1
            if pred_ground is not None:
                vertices, triangles = mcubes.marching_cubes(pred_ground[i], 0.7)
                vertices[:, 0] = (vertices[:, 0] + 0.5) * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['pc_range'][0]
                vertices[:, 1] = (vertices[:, 1] + 0.5) * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['pc_range'][1]
                vertices[:, 2] = (vertices[:, 2] + 0.5) * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['pc_range'][2]
                vertices_ground = vertices.copy()

                vertices_sum = np.concatenate([vertices_object, vertices_ground], axis=0)
            else:
                vertices_sum = vertices_object
           
            mesh_metrics = eval_mesh(vertices_sum, points_occ[i].cpu().numpy().copy())

            '''
            x = np.linspace(0, pred_occ_logits[i].shape[0] - 1, pred_occ_logits[i].shape[0])
            y = np.linspace(0, pred_occ_logits[i].shape[1] - 1, pred_occ_logits[i].shape[1])
            z = np.linspace(0, pred_occ_logits[i].shape[2] - 1, pred_occ_logits[i].shape[2])
            X, Y, Z = np.meshgrid(x, y, z,  indexing='ij')
            vv = np.stack([X, Y, Z], axis=-1)
            v_temp = torch.sigmoid(torch.from_numpy(pred_occ_logits[i])).numpy()
            vv = vv[v_temp >= 0.5]
            vv[:, 0] = vv[:, 0] * (img_metas['pc_range'][3] - img_metas['pc_range'][0]) /  img_metas['occ_size'][0]  + img_metas['occ_size'][0] / 2.0 #+ img_metas['pc_range'][0]
            vv[:, 1] = vv[:, 1] * (img_metas['pc_range'][4] - img_metas['pc_range'][1]) /  img_metas['occ_size'][1]  + img_metas['occ_size'][1] / 2.0 #+ img_metas['pc_range'][1]
            vv[:, 2] = vv[:, 2] * (img_metas['pc_range'][5] - img_metas['pc_range'][2]) /  img_metas['occ_size'][2]  + img_metas['occ_size'][2] / 2.0 #+ img_metas['pc_range'][2]
            '''
     
            return mesh_metrics
            occ_results.append(mesh_metrics)
        return occ_results

    def evaluation_semantic(self, pred, gt, img_metas):
        import open3d as o3d
        color_map = np.array([
            (0, 0, 0),
            (174, 174, 174),  # wall
            (152, 223, 138),  # floor
            (31, 119, 180),  # cabinet
            (255, 187, 120),  # bed
            (188, 189, 34),  # chair
            (140, 86, 75),  # sofa
            (255, 152, 150),  # table
            (214, 39, 40),  # door
            (197, 176, 213),  # window
            (148, 103, 189),  # bookshelf
            (255, 0, 0),  # picture
            (0, 0, 255),  # counter
            (0, 255, 0),
            (255, 255, 0),  # desk
            (66, 188, 102),
            (219, 219, 141),  # curtain
            (140, 57, 197),
            (202, 185, 52),
            (51, 176, 203),
            (200, 54, 131),
            (92, 193, 61),
            (78, 71, 183),
            (172, 114, 82),
            (255, 127, 14),  # refrigerator
            (91, 163, 138),
            (153, 98, 156),
            (140, 153, 101),
            (158, 218, 229),  # shower curtain
            (100, 125, 154),
            (178, 127, 135),
            (120, 185, 128),
            (146, 111, 194),
            (44, 160, 44),  # toilet
            (112, 128, 144),  # sink
            (96, 207, 209),
            (227, 119, 194),  # bathtub
            (213, 92, 176),
            (94, 106, 211),
            (82, 84, 163),  # otherfurn
            (100, 85, 144)
        ])

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

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        return outs['bev_embed'], outs

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, output = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)

        return new_prev_bev, output

def eval_mesh(verts_pred, verts_trgt, threshold=.25, down_sample=.1):
    import open3d as o3d
    """ Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points
    Returns:
        Dict of mesh metrics
    """

    '''
    pcd_pred = o3d.io.read_point_cloud(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    '''
    pcd_pred = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(verts_pred)
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_trgt.points = o3d.utility.Vector3dVector(verts_trgt)

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)
    
    
    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2<threshold).astype('float'))
    recal = np.mean((dist1<threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = np.array([np.mean(dist1),np.mean(dist2),precision,recal,fscore])
    
    return metrics

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1
    
    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    
    """
    import open3d as o3d

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances
