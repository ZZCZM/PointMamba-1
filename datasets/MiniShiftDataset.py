import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from .build import DATASETS # 注册到 PointMamba 的工厂机制中

@DATASETS.register_module()
class MiniShiftDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.data_root = config.DATA_PATH  # 例如：'/root/autodl-tmp/simple3d/MiniShift1/data/MiniShiftAD'
        self.cls = config.CATEGORY         # 例如：'gear' 或 'capsule'
        self.mode = config.MODE            # 'train' 或 'test'
        self.level = getattr(config, 'LEVEL', 'ALL') 
        self.num_points = getattr(config, 'N_POINTS', 16384) 
        
        self.data_path = os.path.join(self.data_root, self.cls, self.mode)
        
        self.pcd_paths = []
        self.gt_paths = []
        self.labels = [] 
        
        self._load_dataset()

    def _load_dataset(self):
        if self.mode == 'train':
            pcd_path = os.path.join(self.data_root, self.cls, 'train', 'good')
            paths = glob.glob(pcd_path + "/*.txt")
            paths.sort()
            self.pcd_paths.extend(paths)
            self.labels.extend([0] * len(paths))
            self.gt_paths.extend([None] * len(paths)) 
            
        elif self.mode == 'test':
            defect_types = os.listdir(self.data_path)
            for defect_type in defect_types:
                if defect_type == 'good':
                    paths = glob.glob(os.path.join(self.data_path, defect_type) + "/*.txt")
                    paths.sort()
                    self.pcd_paths.extend(paths)
                    self.gt_paths.extend([None] * len(paths))
                    self.labels.extend([0] * len(paths))
                else:
                    for lvl in ['easy', 'medium', 'hard']:
                        if self.level == 'ALL' or self.level == lvl:
                            paths = glob.glob(os.path.join(self.data_path, defect_type, lvl) + "/*.txt")
                            paths.sort()
                            gts = [x.replace('test', 'gt') for x in paths]
                            
                            self.pcd_paths.extend(paths)
                            self.gt_paths.extend(gts)
                            self.labels.extend([1] * len(paths))
                            
            assert len(self.pcd_paths) == len(self.gt_paths), "Test paths and GT paths mismatch!"

    def __len__(self):
        return len(self.pcd_paths)

    def __getitem__(self, idx):
        pcd_path = self.pcd_paths[idx]
        label = self.labels[idx]
        gt_path = self.gt_paths[idx]

        unorganized_pc = np.loadtxt(pcd_path, dtype=np.float32)
        
        # 1. 稳健的坐标中心化与缩放 (Robust Normalization)
        unorganized_pc[:, 0] -= np.median(unorganized_pc[:, 0])
        unorganized_pc[:, 1] -= np.median(unorganized_pc[:, 1])
        unorganized_pc[:, 2] -= np.median(unorganized_pc[:, 2])
        
        distances = np.sqrt(np.sum(unorganized_pc**2, axis=1))
        max_dist = np.percentile(distances, 99.9) 
        if max_dist > 0:
            unorganized_pc = unorganized_pc / max_dist

        # 2. 加载真实缺陷 Mask
        if label == 0 or gt_path is None:
            gt = torch.zeros(unorganized_pc.shape[0])
        else:
            gt = np.loadtxt(gt_path)
            gt = torch.tensor(gt).squeeze()
            gt = torch.where(gt > 0.5, 1, 0) 

        # 3. 训练期：致密局部切块 (Local Cropping) 替代 随机降采样
        if self.mode == 'train' and unorganized_pc.shape[0] > self.num_points:
            seed_idx = np.random.randint(unorganized_pc.shape[0])
            seed_point = unorganized_pc[seed_idx]
            
            dist_sq = np.sum((unorganized_pc - seed_point)**2, axis=1)
            nearest_indices = np.argsort(dist_sq)[:self.num_points]
            
            unorganized_pc = unorganized_pc[nearest_indices, :]
            gt = gt[nearest_indices]

        # 4. 返回格式化数据
        pts_tensor = torch.from_numpy(unorganized_pc).float()
        
        if self.mode == 'train':
            return self.cls, pcd_path, pts_tensor
        else:
            ret_dict = {
                'points': pts_tensor,
                'label': torch.tensor(label).long(),
                'point_mask': gt.long(),
                'pcd_path': pcd_path
            }
            return ret_dict