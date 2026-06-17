import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.logger import print_log
from knn_cuda import KNN

def compute_pointwise_anomaly_score(gt_pts, rec_pts, chunk_size=10000):
    """
    计算真实点云与重建点云之间的逐点距离（分块计算防止显存溢出 OOM）
    gt_pts: (B, N, 3) 原始点云 (可能高达 50万点)
    rec_pts: (B, M, 3) 重建的正常流形点云
    """
    from knn_cuda import KNN
    knn_solver = KNN(k=1, transpose_mode=True)
    B, N, _ = gt_pts.shape
    
    # 预先分配最终的分数结果张量
    point_scores = torch.zeros(B, N, device=gt_pts.device)
    
    # 将海量的 N 个点进行分块 (chunk) 计算
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        gt_chunk = gt_pts[:, i:end_idx, :] # 取出一小块点云 (B, chunk_size, 3)
        
        # 计算这小块点到重建流形的最短距离
        dist, _ = knn_solver(rec_pts, gt_chunk) 
        
        # 【修改处】：使用 .reshape(B, -1) 安全对齐维度
        point_scores[:, i:end_idx] = dist.reshape(B, -1)
        
    return point_scores

def test_anomaly(model, test_dataloader, config):
    from utils.logger import print_log
    print_log("Starting Anomaly Detection Evaluation with Ensemble...", logger='PointMamba')
    model.eval()
    
    all_img_scores = []
    all_img_labels = []   
    all_pixel_scores = []
    all_pixel_labels = [] 
    
    # 【核心配置】：对每个零件进行 8 轮不同掩码的重建
    num_ensemble_passes = 8 
    
    with torch.no_grad():
        for idx, batch_data in enumerate(test_dataloader):
            points = batch_data['points'].cuda()
            labels = batch_data['label'].numpy()
            masks_gt = batch_data['point_mask'].numpy()
            
            B, N, _ = points.shape
            target_points = 16384 
            if N > target_points:
                choice = torch.randperm(N, device=points.device)[:target_points]
                points = points[:, choice, :]
                masks_gt = masks_gt[:, choice.cpu().numpy()]

            # ========== 【核心修复：多轮集成推理】 ==========
            ensemble_scores = []
            
            for _ in range(num_ensemble_passes):
                # 每次 forward，模型内部都会生成完全不同的随机掩码
                _, rec_pts = model(points, mode='test')
                rec_pts = rec_pts.reshape(B, -1, 3)
                
                # 计算这一轮的距离分数
                pass_scores = compute_pointwise_anomaly_score(points, rec_pts)
                ensemble_scores.append(pass_scores)
            
            # 将 8 轮的分数堆叠，取平均值 (B, N)
            # 正常区域永远平滑(分数极低)，缺陷区域至少有几次分数极高，均值会被拉大
            point_scores = torch.stack(ensemble_scores, dim=0).mean(dim=0)
            # =================================================
            
            # 样本级打分：取全图均值异常分数最高的 Top 10 个点的平均值
            k = min(10, point_scores.shape[1])
            topk_scores, _ = torch.topk(point_scores, k=k, dim=1)
            img_scores = topk_scores.mean(dim=1).cpu().numpy()
            
            all_img_scores.extend(img_scores)
            all_img_labels.extend(labels)
            
            for i in range(len(labels)):
                all_pixel_scores.extend(point_scores[i].cpu().numpy().flatten())
                all_pixel_labels.extend(masks_gt[i].flatten())

            if (idx + 1) % 10 == 0:
                print_log(f"Evaluated {idx + 1}/{len(test_dataloader)} samples...", logger='PointMamba')

    # 计算最终评估指标
    from sklearn.metrics import roc_auc_score
    img_auroc = roc_auc_score(all_img_labels, all_img_scores)
    print_log(f"=========================================", logger='PointMamba')
    print_log(f"🏆 Image (Sample-level) AUROC: {img_auroc * 100:.2f}%", logger='PointMamba')
    
    if len(all_pixel_labels) > 0 and sum(all_pixel_labels) > 0:
        pixel_auroc = roc_auc_score(all_pixel_labels, all_pixel_scores)
        print_log(f"🎯 Pixel (Point-level) AUROC: {pixel_auroc * 100:.2f}%", logger='PointMamba')
    print_log(f"=========================================", logger='PointMamba')
    
    return img_auroc