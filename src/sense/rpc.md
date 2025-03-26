输入：

```python
{
	'0000': {
		'lidar_np': np.ndarray,
        'ego': np.ndarray([True]),
        'transformation_matrix': np.ndarray,
        'ego_speed': np.ndarray,
        ...
        // 可能还有np.ndarray
	}, 
    '0001': {
		'lidar_np': None,
        'ego': np.ndarray([False]),
        'voxel_features': np.ndarray,  // (7559, 32, 4)
        'voxel_coords': np.ndarray,   // (7559, 3)
        'voxel_num_points': np.ndarray,  // (7559,)
        'transformation_matrix': np.ndarray,
        'ego_speed': np.ndarray,
        ...
        // 可能还有np.ndarray

	},
}
```

输出：

```python
results = {
    'pred_box_tensor': pred_box_tensor.cpu().numpy(),
    'pred_score': pred_score.cpu().numpy(),
    'comm_masks': comm_masks.cpu().numpy(),
    'fused_feat': fused_feat.cpu().numpy()
}
```

