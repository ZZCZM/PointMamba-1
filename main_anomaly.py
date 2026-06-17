import os
import argparse
import datetime
from pathlib import Path

import torch
from utils.config import get_config
from utils.logger import get_logger
from models import build_model_from_cfg
from datasets.MiniShiftDataset import MiniShiftDataset
from tools.runner_anomaly import test_anomaly

def parse_args():
    parser = argparse.ArgumentParser(description='PointMamba Anomaly Detection Test')
    parser.add_argument('--config', type=str, required=True, help='config file path')
    parser.add_argument('--ckpts', type=str, required=True, help='path to pre-trained model weights')
    
    # ======== 为了兼容官方 utils.config 补充的占位参数 ========
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=True)  # 设置为测试模式
    parser.add_argument('--finetune_model', action='store_true', default=False)
    parser.add_argument('--scratch_model', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    # ==========================================================
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 动态补充分布式判定属性，防止底层代码报错
    if not hasattr(args, 'distributed'):
        args.distributed = False

    # ======== 补充实验日志路径，防止 config.py 报错 ========
    args.exp_name = "Anomaly_Test"
    # 创建类似 ./experiments/Anomaly_Test/202405xx-xxxxxx 的目录格式
    args.experiment_path = os.path.join('./experiments', args.exp_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    args.log_name = Path(args.config).stem
    # 确保文件夹存在，用来存放每次测试生成的日志和配置备份
    os.makedirs(args.experiment_path, exist_ok=True)
    # ========================================================

    # 现在传入 args 就不会再报 AttributeError 了
    config = get_config(args, logger=None)
    logger = get_logger("PointMamba")
    
    logger.info(f"Loading testing config from {args.config}")
    
    # 1. 构建测试数据集 (Validation/Test Set)
    test_dataset = MiniShiftDataset(config.dataset.val.others)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, # 测试永远是单张图扫描
        shuffle=False, 
        num_workers=4,
        pin_memory=True, 
        drop_last=False
    )
    logger.info(f"Loaded MiniShift Test Set: {len(test_dataset)} samples.")

    # 2. 构建模型
    model = build_model_from_cfg(config.model)
    model.cuda()
    
    # 3. 加载预训练权重
    logger.info(f"Loading weights from {args.ckpts}")
    checkpoint = torch.load(args.ckpts, map_location='cpu')
    
    # 兼容处理：获取实际的主干网络权重
    if 'base_model' in checkpoint:
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['base_model'].items()}
    elif 'model' in checkpoint:
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    else:
        base_ckpt = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        
    model.load_state_dict(base_ckpt, strict=False)
    
    # 4. 执行异常检测评估
    test_anomaly(model, test_dataloader, config)

if __name__ == '__main__':
    main()