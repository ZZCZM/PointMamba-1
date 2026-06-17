import torch
from knn_cuda import KNN

def compute_pointwise_anomaly_score(gt_pts, rec_pts):
    """
    计算输入点云与重建点云之间的逐点异常分数
    gt_pts: B x N x 3 (真实点云)
    rec_pts: B x N x 3 (模型重建出的正常流形点云)
    """
    # 针对 gt_pts 中的每一个点，在 rec_pts 中寻找最近的 1 个点
    knn_solver = KNN(k=1, transpose_mode=True)
    
    # KNN 返回距离和索引，我们只需要距离
    # dist: B x 1 x N
    dist, _ = knn_solver(rec_pts, gt_pts) 
    
    # 转换为 B x N 的异常热力图向量
    point_anomaly_score = dist.squeeze(1) 
    
    return point_anomaly_score