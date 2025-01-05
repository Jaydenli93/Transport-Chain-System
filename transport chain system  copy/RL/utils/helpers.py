import torch
import numpy as np
import random
from typing import Optional
from datetime import datetime
import os
import json
from typing import Any

def set_random_seed(seed: int, 
                   env: Optional[Any] = None) -> None:
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if env is not None:
        env.seed(seed)

def create_experiment_dir(base_dir: str) -> str:
    """创建实验目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def save_config(config: Any, path: str) -> None:
    """保存配置"""
    with open(path, 'w') as f:
        json.dump(config.__dict__, f, indent=4)