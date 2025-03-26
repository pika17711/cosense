
from typing import Any
from config import CONFIG

from server import agent_server

class BaseDataRetriever:

    def __init__(self) -> None:
        pass

    async def get_ego_data(self):
        """
        return:
        {
            'lidar_np': np.ndarray,
            'is_ego': np.ndarray([True]),
            'transformation_matrix': np.ndarray,
            'ego_speed': np.ndarray,
            ...
        }
        """
        data_dict = {
            "is_ego": True,
            "lidar_np": await agent_server.wait_ego_pcd(),
            "params": None
        }
        return data_dict

    async def get_other_data(self):
        """
        return: 
        {
            "is_ego": np.ndarray([True]),
            'voxel_features': np.ndarray,  // (7559, 32, 4)
            'voxel_coords': np.ndarray,   // (7559, 3)
            'voxel_num_points': np.ndarray,  // (7559,)
            "ego_pose": pose,
            "lidar_pose": pose,
            "transformation_matrix" : matrix, // lidar to ego
            "speed": speed,
            ...
        }
        """
        return await agent_server.get_neighbors_cached_data()

    async def __call__(self) -> Any:

        data_dict = await agent_server.get_neighbors_cached_data()
        data_dict[CONFIG['id']] = await self.get_ego_data()

        return data_dict