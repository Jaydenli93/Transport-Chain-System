class RLError(Exception):
    """RL模块基础异常类"""
    pass

class TrainingError(RLError):
    """训练过程异常"""
    pass

class EnvironmentError(RLError):
    """环境相关异常"""
    pass

class ModelError(RLError):
    """模型相关异常"""
    pass