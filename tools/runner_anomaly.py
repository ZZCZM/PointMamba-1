import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.logger import print_log
from knn_cuda import KNN

def extract_overlapping_blocks(points, target_points=16384, overlap_factor=1.5):
    """
    基于原生 PyTorch 矩阵运算的重叠切块提取机制 (极大加速版)
    points: (1, N, 3) 原始完整点云
    target_points: 每个局部块的点数 (默认 16384)
    overlap_factor: 重叠因子，值越大提取的块越多，覆盖越密
    返回: blocks_idx (num_blocks, target_points) 索引张量
    """
    N = points.shape[1]
    # 计算需要切多少个块才能覆盖全图，并加上重叠因子
    num_blocks = int((N / target_points) * overlap_factor)
    num_blocks = max(1, num_blocks) # 至少有1个块
    
    # 1. 随机选取 num_blocks 个种子点
    seed_indices = torch.randperm(N, device=points.device)[:num_blocks]
    seeds = points[:, seed_indices, :] # (1, num_blocks, 3)
    
    # =========================================================================
    # 【核心加速优化】：废弃 knn_cuda，改用 PyTorch 原生张量广播计算距离
    # points: (1, N, 3) -> 扩展为 (1, 1, N, 3)
    # seeds: (1, num_blocks, 3) -> 扩展为 (1, num_blocks, 1, 3)
    # 计算每个种子点到全图 50 万个点的欧式距离平方，利用 GPU 矩阵乘法极速完成
    # =========================================================================
    dist_sq = torch.sum((points.unsqueeze(1) - seeds.unsqueeze(2)) ** 2, dim=-1) # 结果形状: (1, num_blocks, N)
    
    # 2. 找到距离最近的 target_points 个点 (等效于 KNN，但快无数倍)
    _, block_indices = torch.topk(dist_sq, k=target_points, dim=-1, largest=False)
    
    return block_indices.squeeze(0) # (num_blocks, target_points)

def test_anomaly(model, test_dataloader, config):
    print_log("Starting Block-Cropping Anomaly Detection Evaluation...", logger='PointMamba')
    model.eval()
    
    all_img_scores = []
    all_img_labels = []   
    all_pixel_scores = []
    all_pixel_labels = [] 
    
    target_points = 16384 
    
    # 实例化一个轻量级的 KNN 求解器，用于局部块内的对齐 (此处 k=1，计算极快，保留)
    local_knn_solver = KNN(k=1, transpose_mode=True)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(test_dataloader):
            points = batch_data['points'].cuda() # (1, N, 3)
            labels = batch_data['label'].numpy() # (1,)
            masks_gt = batch_data['point_mask'].numpy() # (1, N)
            
            B, N, C = points.shape
            
            # 初始化全局计分板：用于累加每个点的异常分数和被计算的次数
            global_anomaly_scores = torch.zeros(B, N, device=points.device)
            global_counts = torch.zeros(B, N, device=points.device)

            # ======== 1. 获取重叠局部块的索引 ========
            if N > target_points:
                block_indices = extract_overlapping_blocks(points, target_points=target_points, overlap_factor=2.0)
            else:
                # 如果点数少于 16384，直接作为 1 个块
                block_indices = torch.arange(N, device=points.device).unsqueeze(0)
            
            num_blocks = block_indices.shape[0]
            
            # ======== 2. 遍历每个局部块进行推理 ========
            for b_idx in range(num_blocks):
                # 获取当前块的点云 (1, 16384, 3)
                curr_idx = block_indices[b_idx]
                block_pts = points[:, curr_idx, :] 
                
                # 【接口修改】接收模型返回的 3 个变量 (原始绝对坐标, 重建绝对坐标, 偏移量)
                _, rec_pts, _ = model(block_pts, mode='test')
                
                # 将重建的点云展平为 (1, 16384, 3)
                rec_pts = rec_pts.reshape(B, -1, 3)
                
                # 【核心对齐】由于模型内部存在 FPS 采样和 Hilbert 曲线乱序，直接相减会导致错位。
                # 由于切块后点数仅 1.6 万，在此处进行局部的 KNN 距离计算既不会 OOM，又能完美映射偏移距离。
                dist, _ = local_knn_solver(rec_pts, block_pts) 
                
                # 获取该局部块内每个点的异常分数 (1, 16384)
                offset_scores = dist.squeeze(-1)
                
                # 将该块的分数累加回全局计分板
                global_anomaly_scores[:, curr_idx] += offset_scores
                global_counts[:, curr_idx] += 1
                
            # ======== 3. 汇总与平均全局分数 ========
            # 防止有些点没被采样到(除以0)，加上 epsilon
            final_point_scores = global_anomaly_scores / (global_counts + 1e-8) # (1, N)
            
            # 样本级打分：取全图最高 Top 1% 的点的平均值 (比单纯 Top 10 更稳健)
            k = max(10, int(N * 0.01)) 
            topk_scores, _ = torch.topk(final_point_scores, k=k, dim=1)
            img_score = topk_scores.mean(dim=1).cpu().numpy()
            
            all_img_scores.extend(img_score)
            all_img_labels.extend(labels)
            
            # 记录逐点分数用于计算 Pixel AUROC
            for i in range(len(labels)):
                all_pixel_scores.extend(final_point_scores[i].cpu().numpy().flatten())
                all_pixel_labels.extend(masks_gt[i].flatten())

            # =========================================================================
            # 【日志体验优化】：把 % 10 改为 % 1，每个样本扫描完立即打印，消除“卡死”错觉
            # =========================================================================
            if (idx + 1) % 1 == 0:
                print_log(f"Evaluated {idx + 1}/{len(test_dataloader)} samples...", logger='PointMamba')

    # 计算最终评估指标
    img_auroc = roc_auc_score(all_img_labels, all_img_scores)
    print_log(f"=========================================", logger='PointMamba')
    print_log(f"🏆 Image (Sample-level) AUROC: {img_auroc * 100:.2f}%", logger='PointMamba')
    
    if len(all_pixel_labels) > 0 and sum(all_pixel_labels) > 0:
        pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
        print_log(f"🎯 Pixel (Point-level) AUROC: {pixel_auroc * 100:.2f}%", logger='PointMamba')
    print_log(f"=========================================", logger='PointMamba')
    
    return img_auroc