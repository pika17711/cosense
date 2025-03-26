
from server import CONFIG


class CommDecisionEngine:
    """通信决策引擎"""
    def __init__(self, threshold=0.5):
        self.conf_threshold = threshold

    def select_targets(self, conf_map):
        """选择需要通信的目标节点"""
        # 生成通信需求矩阵
        
        # 查找满足条件的邻居节点
        target_neighbors = []
        for neighbor in self._get_neighbors():
            overlap = self._calculate_overlap(conf_map, neighbor.req_map)
            if overlap > CONFIG["processing"]["overlap_threshold"]:
                target_neighbors.append(neighbor)

        return target_neighbors

    def _calculate_overlap(self, conf_map, neighbor_req_map):
        ...
