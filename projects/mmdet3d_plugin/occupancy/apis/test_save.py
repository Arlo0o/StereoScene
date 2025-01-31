import os.path as osp
import pickle
import shutil
import tempfile
import time
import os

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core import encode_mask_results

import mmcv
import numpy as np
import pycocotools.mask as mask_util
import pdb
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from projects.mmdet3d_plugin.utils.formating import cm_to_ious, format_results
from projects.mmdet3d_plugin.utils.ssc_metric import SSCMetrics
from projects.mmdet3d_plugin.utils.semkitti_io import get_inv_map

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def save_output_semantic_kitti(output_voxels, save_path, 
                    sequence_id, frame_id):
    
    output_voxels = torch.argmax(output_voxels, dim=0)
    output_voxels = output_voxels.cpu().numpy().reshape(-1)
    # remap to lidarseg ID
    inv_map = get_inv_map()
    output_voxels = inv_map[output_voxels].astype(np.uint16)
    
    save_folder = "{}/sequences/{}/predictions".format(save_path, sequence_id)
    save_file = os.path.join(save_folder, "{}.label".format(frame_id))
    os.makedirs(save_folder, exist_ok=True)
    
    with open(save_file, 'wb') as f:
        output_voxels.tofile(f)
        print('\n save to {}'.format(save_file))

def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, 
        show_score_thr=0.3, test_save=None):
    
    model.eval()
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    # evaluate ssc
    is_semkitti = hasattr(dataset, 'camera_used')
    ssc_metric = SSCMetrics().cuda()
    logger.info(parameter_count_table(model, max_depth=4))
    
    batch_size = 1
    
    for i, data in enumerate(data_loader):
 
            myout={}
            with torch.no_grad():
                    
                result = model(return_loss=False, rescale=True, **data)
                    

                # KITTI workaround
                if not isinstance(result, dict):
                    prog_bar.update()
                    continue
                
            img_metas = data['img_metas'].data[0][0]
           
             
            filename = img_metas['img_filename'][0].split('.')[-2].split('/')[-1]
 
            
            output_voxels = torch.argmax(result['output_voxels'], dim=1)
            target_voxels = result['target_voxels'].clone()

            myout["output_voxels"] = output_voxels.squeeze(0).cpu().detach()
            myout["target_voxels"] = target_voxels.squeeze(0).cpu().detach()


            outputs = {"singleout": myout}
            mmcv.dump(outputs, "./_pickle_final/"+ filename + ".pkl")
            
    res = {"out": myout}
    return res

def custom_multi_gpu_test(model, data_loader, tmpdir=None,
        gpu_collect=False, test_save=None):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        
    ssc_results = []
    ssc_metric = SSCMetrics().cuda()
    
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    logger = get_root_logger()
    logger.info(parameter_count_table(model))
    
    is_test_submission = test_save is not None
    if is_test_submission:
        os.makedirs(test_save, exist_ok=True)
    
    # evaluate lidarseg
    evaluation_semantic = 0
    
    batch_size = 1
    for i, data in enumerate(data_loader):
        
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            
        # nusc lidar segmentation
        if 'evaluation_semantic' in result:
            evaluation_semantic += result['evaluation_semantic']
        
        # occupancy prediction
        if is_test_submission:
            img_metas = data['img_metas'].data[0][0]
            assert result['output_voxels'].shape[0] == 1
            save_output_semantic_kitti(result['output_voxels'][0], 
                test_save, img_metas['sequence'], img_metas['frame_id'])
        
        else:
            ssc_results_i = ssc_metric.compute_single(
                y_pred=torch.argmax(result['output_voxels'], dim=1), 
                y_true=result['target_voxels'],
            )
            ssc_results.append(ssc_results_i)
            # ssc_results.append( [ [result['output_voxels']], [torch.argmax(result['output_voxels'], dim=1)], [result['target_voxels']]  ]  )
        print(img_metas['sequence'], img_metas['frame_id'])         
        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    if is_test_submission:
        return None
    
    res = {}
    res['ssc_results'] = collect_results_cpu(ssc_results, len(dataset), tmpdir)
    
    if type(evaluation_semantic) is np.ndarray:
        # convert to tensor for reduce_sum
        evaluation_semantic = torch.from_numpy(evaluation_semantic).cuda()
        dist.all_reduce(evaluation_semantic, op=dist.ReduceOp.SUM)
        res['evaluation_semantic'] = evaluation_semantic.cpu().numpy()
    
    return res

def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    
    # collect all parts
    if rank != 0:
        return None
    
    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    
    # sort the results
    if type == 'list':
        ordered_results = []
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
    
    else:
        raise NotImplementedError
    
 
    
    return ordered_results