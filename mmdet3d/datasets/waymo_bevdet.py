# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
import tempfile
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
import torch
import copy
from mmcv.utils import print_log
from os import path as osp

from mmdet.datasets import DATASETS
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .kitti_dataset import KittiDataset


@DATASETS.register_module()
class waymoDatasetForBevdet(KittiDataset):
    """Waymo Dataset.

    This class serves as the API for experiments on the Waymo Dataset.

    Please refer to `<https://waymo.com/open/download/>`_for data downloading.
    It is recommended to symlink the dataset root to $MMDETECTION3D/data and
    organize them as the doc shows.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': box in LiDAR coordinates
            - 'Depth': box in depth coordinates, usually for indoor dataset
            - 'Camera': box in camera coordinates
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list): The range of point cloud used to filter
            invalid predicted boxes. Default: [-85, -85, -5, 85, 85, 5].
    """

    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 img_info_prototype='bevdet',
                 data_config=None,
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 speed_mode='abs_dis',
                 max_interval=3,
                 min_interval=0,
                 test_adj_ids=None,
                 test_adj='prev',
                 fix_direction=False,
                 ):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            split=split,
            pts_prefix=pts_prefix,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            pcd_limit_range=pcd_limit_range)

        # to load a subset, just set the load_interval in the dataset config
        self.data_infos = self.data_infos[::load_interval]
        self.img_info_prototype = img_info_prototype
        self.data_config = data_config
        self.speed_mode = speed_mode
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.test_adj = test_adj
        self.test_adj_ids = test_adj_ids
        self.fix_direction = fix_direction
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

    def _get_pts_filename(self, idx):
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:07d}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Standard input_dict consists of the
                data information.

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - img_prefix (str | None): prefix of image files
                - img_info (dict): image info
                - lidar2img (list[np.ndarray], optional): transformations from
                    lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']


        # TODO: consider use torch.Tensor only
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c_0 = info['calib']['Tr_velo_to_cam_0'].astype(np.float32)
        Trv2c_1 = info['calib']['Tr_velo_to_cam_1'].astype(np.float32)
        Trv2c_2 = info['calib']['Tr_velo_to_cam_2'].astype(np.float32)
        Trv2c_3 = info['calib']['Tr_velo_to_cam_3'].astype(np.float32)
        Trv2c_4 = info['calib']['Tr_velo_to_cam_4'].astype(np.float32)

        P0 = info['calib']['P0'].astype(np.float32)
        P1 = info['calib']['P1'].astype(np.float32)
        P2 = info['calib']['P2'].astype(np.float32)
        P3 = info['calib']['P3'].astype(np.float32)
        P4 = info['calib']['P4'].astype(np.float32)
        # lidar2img = P0 @ rect @ Trv2c

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_info=dict(),
            img_prefix=None)
            # lidar2img=lidar2img)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            # gt_boxes = annos['gt_bboxes_3d']
            #
            # gt_labels = annos['gt_labels_3d']
            # annos_gt = (gt_boxes,gt_labels)
            #
            # input_dict['ann_infos'] = annos_gt
            input_dict['ann_info'] = annos

        intrincs_dy={
            'CAM_FRONT' : P0,
        'CAM_FRONT_LEFT' : P1,
        'CAM_FRONT_RIGHT': P2,
             'SIDE_LEFT': P3,
            'SIDE_RIGHT': P4,
        }

        Trv2c_dy={
            'CAM_FRONT' : Trv2c_0,
        'CAM_FRONT_LEFT' : Trv2c_1,
        'CAM_FRONT_RIGHT': Trv2c_2,
             'SIDE_LEFT': Trv2c_3,
            'SIDE_RIGHT': Trv2c_4,
        }



        if self.img_info_prototype == 'bevdet':
            cams = self.data_config['cams']
            for cam in cams:
                camera2lidar = np.linalg.inv(rect @ Trv2c_dy[cam])
                img_filename = os.path.join(self.data_root,
                                            info['image'][cam]['image_path'])
                input_dict['img_info'][cam] = dict(data_path=img_filename,
                                                           cam_intrinsic=intrincs_dy[cam],
                                                           sensor2lidar_rotation=camera2lidar[:3, :3],
                                                           sensor2lidar_translation=camera2lidar[:3, 3])
            if not self.test_mode:
                bbox = input_dict['ann_info']['gt_bboxes_3d'].tensor
                bbox = torch.cat([bbox, torch.zeros((bbox.shape[0], 2))], dim=-1)
                input_dict['ann_info']['gt_bboxes_3d'] = \
                    LiDARInstance3DBoxes(tensor=bbox, box_dim=bbox.shape[-1], with_yaw=input_dict['ann_info']['gt_bboxes_3d'].with_yaw)

        return input_dict
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam_0'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        # convert gt_bboxes_3d to velodyne coordinates
        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))
        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results

    # def format_results(self,
    #                    outputs,
    #                    pklfile_prefix=None,
    #                    submission_prefix=None,
    #                    data_format='waymo'):
    #     """Format the results to pkl file.
    #
    #     Args:
    #         outputs (list[dict]): Testing results of the dataset.
    #         pklfile_prefix (str | None): The prefix of pkl files. It includes
    #             the file path and the prefix of filename, e.g., "a/b/prefix".
    #             If not specified, a temp file will be created. Default: None.
    #         submission_prefix (str | None): The prefix of submitted files. It
    #             includes the file path and the prefix of filename, e.g.,
    #             "a/b/prefix". If not specified, a temp file will be created.
    #             Default: None.
    #         data_format (str | None): Output data format. Default: 'waymo'.
    #             Another supported choice is 'kitti'.
    #
    #     Returns:
    #         tuple: (result_files, tmp_dir), result_files is a dict containing
    #             the json filepaths, tmp_dir is the temporal directory created
    #             for saving json files when jsonfile_prefix is not specified.
    #     """
    #     if pklfile_prefix is None:
    #         tmp_dir = tempfile.TemporaryDirectory()
    #         pklfile_prefix = osp.join(tmp_dir.name, 'results')
    #     else:
    #         tmp_dir = None
    #
    #     assert ('waymo' in data_format or 'kitti' in data_format), \
    #         f'invalid data_format {data_format}'
    #
    #     if (not isinstance(outputs[0], dict)) or 'img_bbox' in outputs[0]:
    #         raise TypeError('Not supported type for reformat results.')
    #     elif 'pts_bbox' in outputs[0]:
    #         result_files = dict()
    #         for name in outputs[0]:
    #             results_ = [out[name] for out in outputs]
    #             pklfile_prefix_ = pklfile_prefix + name
    #             if submission_prefix is not None:
    #                 submission_prefix_ = f'{submission_prefix}_{name}'
    #             else:
    #                 submission_prefix_ = None
    #             result_files_ = self.bbox2result_kitti(results_, self.CLASSES,
    #                                                    pklfile_prefix_,
    #                                                    submission_prefix_)
    #             result_files[name] = result_files_
    #     else:
    #         result_files = self.bbox2result_kitti(outputs, self.CLASSES,
    #                                               pklfile_prefix,
    #                                               submission_prefix)
    #     if 'waymo' in data_format:
    #         from ..core.evaluation.waymo_utils.prediction_kitti_to_waymo import \
    #             KITTI2Waymo  # noqa
    #         waymo_root = osp.join(
    #             self.data_root.split('kitti_format')[0], 'waymo_format')
    #         if self.split == 'training':
    #             waymo_tfrecords_dir = osp.join(waymo_root, 'validation')
    #             prefix = '1'
    #         elif self.split == 'testing':
    #             waymo_tfrecords_dir = osp.join(waymo_root, 'testing')
    #             prefix = '2'
    #         else:
    #             raise ValueError('Not supported split value.')
    #         save_tmp_dir = tempfile.TemporaryDirectory()
    #         waymo_results_save_dir = save_tmp_dir.name
    #         waymo_results_final_path = f'{pklfile_prefix}.bin'
    #         if 'pts_bbox' in result_files:
    #             converter = KITTI2Waymo(result_files['pts_bbox'],
    #                                     waymo_tfrecords_dir,
    #                                     waymo_results_save_dir,
    #                                     waymo_results_final_path, prefix)
    #         else:
    #             converter = KITTI2Waymo(result_files, waymo_tfrecords_dir,
    #                                     waymo_results_save_dir,
    #                                     waymo_results_final_path, prefix)
    #         converter.convert()
    #         save_tmp_dir.cleanup()
    #
    #     return result_files, tmp_dir


    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.data_infos[sample_id],
                                       self.speed_mode, self.img_info_prototype,
                                       self.max_interval, self.test_adj,
                                       self.fix_direction,
                                       self.test_adj_ids)
            sample_index = self.data_infos[sample_id]['image']['image_idx']

            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                nusc_anno = dict(
                    sample_index=sample_index,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score)
                annos.append(nusc_anno)
            nusc_annos[sample_index] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path


    def evaluate(self,
                 results,
                 metric='waymo',
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        assert ('waymo' in metric or 'kitti' in metric), \
            f'invalid metric {metric}'
        if 'kitti' in metric:
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='kitti')
            from mmdet3d.core.evaluation import kitti_eval
            gt_annos = [info['annos'] for info in self.data_infos]

            if isinstance(result_files, dict):
                ap_dict = dict()
                for name, result_files_ in result_files.items():
                    eval_types = ['bev', '3d']
                    ap_result_str, ap_dict_ = kitti_eval(
                        gt_annos,
                        result_files_,
                        self.CLASSES,
                        eval_types=eval_types)
                    for ap_type, ap in ap_dict_.items():
                        ap_dict[f'{name}/{ap_type}'] = float(
                            '{:.4f}'.format(ap))

                    print_log(
                        f'Results of {name}:\n' + ap_result_str, logger=logger)

            else:
                ap_result_str, ap_dict = kitti_eval(
                    gt_annos,
                    result_files,
                    self.CLASSES,
                    eval_types=['bev', '3d'])
                print_log('\n' + ap_result_str, logger=logger)
        if 'waymo' in metric:
            waymo_root = osp.join(
                self.data_root.split('kitti_format')[0], 'waymo_format')
            if pklfile_prefix is None:
                eval_tmp_dir = tempfile.TemporaryDirectory()
                pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
            else:
                eval_tmp_dir = None
            result_files, tmp_dir = self.format_results(
                results,
                pklfile_prefix,
                submission_prefix,
                data_format='waymo')
            import subprocess
            ret_bytes = subprocess.check_output(
                'mmdet3d/core/evaluation/waymo_utils/' +
                f'compute_detection_metrics_main {pklfile_prefix}.bin ' +
                f'{waymo_root}/gt.bin',
                shell=True)
            ret_texts = ret_bytes.decode('utf-8')
            print_log(ret_texts)
            # parse the text to get ap_dict
            ap_dict = {
                'Vehicle/L1 mAP': 0,
                'Vehicle/L1 mAPH': 0,
                'Vehicle/L2 mAP': 0,
                'Vehicle/L2 mAPH': 0,
                'Pedestrian/L1 mAP': 0,
                'Pedestrian/L1 mAPH': 0,
                'Pedestrian/L2 mAP': 0,
                'Pedestrian/L2 mAPH': 0,
                'Sign/L1 mAP': 0,
                'Sign/L1 mAPH': 0,
                'Sign/L2 mAP': 0,
                'Sign/L2 mAPH': 0,
                'Cyclist/L1 mAP': 0,
                'Cyclist/L1 mAPH': 0,
                'Cyclist/L2 mAP': 0,
                'Cyclist/L2 mAPH': 0,
                'Overall/L1 mAP': 0,
                'Overall/L1 mAPH': 0,
                'Overall/L2 mAP': 0,
                'Overall/L2 mAPH': 0
            }
            mAP_splits = ret_texts.split('mAP ')
            mAPH_splits = ret_texts.split('mAPH ')
            for idx, key in enumerate(ap_dict.keys()):
                split_idx = int(idx / 2) + 1
                if idx % 2 == 0:  # mAP
                    ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
                else:  # mAPH
                    ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
            ap_dict['Overall/L1 mAP'] = \
                (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
                 ap_dict['Cyclist/L1 mAP']) / 3
            ap_dict['Overall/L1 mAPH'] = \
                (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
                 ap_dict['Cyclist/L1 mAPH']) / 3
            ap_dict['Overall/L2 mAP'] = \
                (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
                 ap_dict['Cyclist/L2 mAP']) / 3
            ap_dict['Overall/L2 mAPH'] = \
                (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
                 ap_dict['Cyclist/L2 mAPH']) / 3
            if eval_tmp_dir is not None:
                eval_tmp_dir.cleanup()

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return ap_dict

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert results to kitti format for evaluation and test submission.

        Args:
            net_outputs (List[np.ndarray]): list of array storing the
                bbox and score
            class_nanes (List[String]): A list of class names
            pklfile_prefix (str | None): The prefix of pkl file.
            submission_prefix (str | None): The prefix of submission file.

        Returns:
            List[dict]: A list of dict have the kitti 3d format
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to KITTI format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            image_shape = info['image']['image_shape'][:2]

            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                anno = {
                    'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation_y': [],
                    'score': []
                }

                for box, box_lidar, bbox, score, label in zip(
                        box_preds, box_preds_lidar, box_2d_preds, scores,
                        label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(
                        -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6])
                    anno['bbox'].append(bbox)
                    anno['dimensions'].append(box[3:6])
                    anno['location'].append(box[:3])
                    anno['rotation_y'].append(box[6])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)

                if submission_prefix is not None:
                    curr_file = f'{submission_prefix}/{sample_idx:07d}.txt'
                    with open(curr_file, 'w') as f:
                        bbox = anno['bbox']
                        loc = anno['location']
                        dims = anno['dimensions']  # lhw -> hwl

                        for idx in range(len(bbox)):
                            print(
                                '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} '
                                '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.
                                format(anno['name'][idx], anno['alpha'][idx],
                                       bbox[idx][0], bbox[idx][1],
                                       bbox[idx][2], bbox[idx][3],
                                       dims[idx][1], dims[idx][2],
                                       dims[idx][0], loc[idx][0], loc[idx][1],
                                       loc[idx][2], anno['rotation_y'][idx],
                                       anno['score'][idx]),
                                file=f)
            else:
                annos.append({
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                })
            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the boxes into valid format.

        Args:
            box_dict (dict): Bounding boxes to be converted.

                - boxes_3d (:obj:``LiDARInstance3DBoxes``): 3D bounding boxes.
                - scores_3d (np.ndarray): Scores of predicted boxes.
                - labels_3d (np.ndarray): Class labels of predicted boxes.
            info (dict): Dataset information dictionary.

        Returns:
            dict: Valid boxes after conversion.

                - bbox (np.ndarray): 2D bounding boxes (in camera 0).
                - box3d_camera (np.ndarray): 3D boxes in camera coordinates.
                - box3d_lidar (np.ndarray): 3D boxes in lidar coordinates.
                - scores (np.ndarray): Scores of predicted boxes.
                - label_preds (np.ndarray): Class labels of predicted boxes.
                - sample_idx (np.ndarray): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        P0 = info['calib']['P0'].astype(np.float32)
        P0 = box_preds.tensor.new_tensor(P0)

        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        box_corners_in_image = points_cam2img(box_corners, P0)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx,
            )
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx,
            )

def output_to_nusc_box(detection, info, speed_mode,
                       img_info_prototype, max_interval, test_adj, fix_direction,
                       test_adj_ids):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    velocity_all = box3d.tensor[:, 7:9]
    if img_info_prototype =='bevdet_sequential':
        if info['prev'] is None or info['next'] is None:
            adjacent = 'prev' if info['next'] is None else 'next'
        else:
            adjacent = test_adj
        if adjacent == 'next' and not fix_direction:
            velocity_all = -velocity_all
        if type(info[adjacent]) is list:
            select_id = min(max_interval // 2, len(info[adjacent]) - 1)
            # select_id = min(2, len(info[adjacent]) - 1)
            info_adj = info[adjacent][select_id]
        else:
            info_adj = info[adjacent]
        if 'dis' in speed_mode and test_adj_ids is None:
            time = abs(1e-6 * info['timestamp'] - 1e-6 * info_adj['timestamp'])
            velocity_all = velocity_all / time

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*velocity_all[i,:], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list