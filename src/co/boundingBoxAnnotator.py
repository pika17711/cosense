
import logging


class BoundingBoxAnnotator:
    """目标标注模块（新增类）"""
    def __init__(self, model_path="detection_model.pth"):
        self.class_names = ['car', 'pedestrian', 'cyclist']
    
    def __call__(self, result):
        pred_box_tensor = result['pred_box_tensor']
        pred_score = ['pred_score']
        logging.log(logging.DEBUG, f'annotate result {pred_box_tensor.shape} {pred_score.shape}')