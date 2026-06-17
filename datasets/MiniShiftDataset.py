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
        self.level = getattr(config, 'LEVEL', 'ALL') # 'easy', 'medium', 'hard' 或 'ALL'
        self.num_points = getattr(config, 'N_POINTS', 16384) # 训练时的截断/采样点数
        
        self.data_path = os.path.join(self.data_root, self.cls, self.mode)
        
        self.pcd_paths = []
        self.gt_paths = []
        self.labels = [] # good: 0, anomaly: 1
        
        self._load_dataset()

    def _load_dataset(self):
        if self.mode == 'train':
            # 训练集：只加载 good 的正常样本
            pcd_path = os.path.join(self.data_root, self.cls, 'train', 'good')
            paths = glob.glob(pcd_path + "/*.txt")
            paths.sort()
            self.pcd_paths.extend(paths)
            self.labels.extend([0] * len(paths))
            self.gt_paths.extend([None] * len(paths)) # 正常样本没有异常 mask
            
        elif self.mode == 'test':
            # 测试集：加载 good 样本以及各个难度级别的缺陷样本
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
                            # MiniShift 提供的规则：gt 路径由 test 替换而来
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

        # 1. 加载点云坐标
        unorganized_pc = np.loadtxt(pcd_path, dtype=np.float32)
        
        # 2. 坐标中心化 (Centering)
        unorganized_pc[:, 0] -= np.mean(unorganized_pc[:, 0])
        unorganized_pc[:, 1] -= np.mean(unorganized_pc[:, 1])
        unorganized_pc[:, 2] -= np.mean(unorganized_pc[:, 2])
        
        # 3. 缩放归一化 (Scaling) -> 对 Mamba 收敛极其重要！
        max_dist = np.max(np.sqrt(np.sum(unorganized_pc**2, axis=1)))
        if max_dist > 0:
            unorganized_pc = unorganized_pc / max_dist

        # 4. 加载真实缺陷 Mask
        if label == 0 or gt_path is None:
            gt = torch.zeros(unorganized_pc.shape[0])
        else:
            gt = np.loadtxt(gt_path)
            gt = torch.tensor(gt).squeeze()
            gt = torch.where(gt > 0.5, 1, 0) # 1为缺陷区域

        # 5. 训练期的降采样防护
        if self.mode == 'train' and unorganized_pc.shape[0] > self.num_points:
            choice = np.random.choice(unorganized_pc.shape[0], self.num_points, replace=False)
            unorganized_pc = unorganized_pc[choice, :]
            gt = gt[choice]

        # 6. 根据不同模式返回对应格式
        pts_tensor = torch.from_numpy(unorganized_pc).float()
        
        if self.mode == 'train':
            # 【兼容官方】预训练脚本需要 3 个变量：(taxonomy_ids, model_ids, data)
            # 我们把类别名和文件路径作为前两个 ID，真正的点云张量作为第三个
            return self.cls, pcd_path, pts_tensor
        else:
            # 【兼容我们自己的测试】异常检测评估需要详尽的标签和 Mask 字典
            ret_dict = {
                'points': pts_tensor,
                'label': torch.tensor(label).long(),
                'point_mask': gt.long(),
                'pcd_path': pcd_path
            }
            return ret_dict