
import numpy as np
from scipy.ndimage import zoom

def transform_communication_mask(mask, target_h, target_w):
    """
    调整通信掩码的高度和宽度以匹配给定的目标尺寸。

    参数:
        mask (np.ndarray): 输入通信掩码，形状为 (H, W)。
        target_h (int): 目标高度。
        target_w (int): 目标宽度。

    返回:
        np.ndarray: 调整后的通信掩码，形状为 (target_h, target_w)。
    """
    H, W = mask.shape
    zoom_factors = (target_h / H, target_w / W)
    resized_mask = zoom(mask, zoom_factors)
    return resized_mask


mask = np.array([[1, 0, 0],
                 [0, 1, 1],
                 [0, 1, 0]])
print(mask)
m2 = transform_communication_mask(mask, 10, 10)
print(m2)