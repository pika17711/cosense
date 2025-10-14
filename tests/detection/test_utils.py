import torch
from utils.common import load_yaml
from opencood.visualization.vis_utils import bbx2oabb
import numpy as np


def get_oabbs_gt(shared_info, left_hand_coordinate=True):
    hypes = shared_info.get_hypes()
    post_processor = shared_info.get_post_processor()

    object_stack = []
    object_id_stack = []

    paths = ['D:\\Documents\\datasets\\OPV2V\\test_tmp\\two\\2021_09_03_09_32_17\\311\\006220',
             'D:\\Documents\\datasets\\OPV2V\\test_tmp\\two\\2021_09_03_09_32_17\\302\\006220']

    for path in paths:
        yaml_load = load_yaml(path + '.yaml')
        with shared_info.post_processor_lock:
            object_bbx_center, object_bbx_mask, object_ids = \
                post_processor.generate_object_center([{'params': yaml_load}], shared_info.get_lidar_pose_copy())

        object_bbx_center = object_bbx_center[object_bbx_mask == 1]
        object_stack.append(object_bbx_center)
        object_id_stack += object_ids

        # exclude all repetitive objects
    unique_indices = [object_id_stack.index(x) for x in set(object_id_stack)]
    object_stack = np.vstack(object_stack)
    object_stack = object_stack[unique_indices]

    # make sure bounding boxes across all frames have the same number
    object_bbx_center = \
        np.zeros((hypes['postprocess']['max_num'], 7))
    mask = np.zeros(hypes['postprocess']['max_num'])
    object_bbx_center[:object_stack.shape[0], :] = object_stack
    mask[:object_stack.shape[0]] = 1
    object_bbx_mask = mask

    # generate the anchor boxes
    with shared_info.post_processor_lock:
        anchor_box = post_processor.generate_anchor_box()

    # generate targets label
    with shared_info.post_processor_lock:
        label_dict = post_processor.generate_label(gt_box_center=object_bbx_center,
                                                   anchors=anchor_box,
                                                   mask=mask)
    object_ids = [object_id_stack[i] for i in unique_indices]

    object_bbx_center = torch.from_numpy(np.array([object_bbx_center]))
    object_bbx_mask = torch.from_numpy(np.array([object_bbx_mask]))

    label_dict_list = [label_dict]
    with shared_info.post_processor_lock:
        label_dict = post_processor.collate_batch(label_dict_list)

    transformation_matrix = torch.eye(4)

    with shared_info.post_processor_lock:
        gt_box_tensor = post_processor.generate_gt_bbx({'ego': {'object_bbx_center': object_bbx_center,
                                                                'object_bbx_mask': object_bbx_mask,
                                                                'object_ids': object_ids,
                                                                'transformation_matrix': transformation_matrix}})
    oabbs_gt = bbx2oabb(gt_box_tensor, color=(0, 1, 0), left_hand_coordinate=left_hand_coordinate)
    return oabbs_gt
