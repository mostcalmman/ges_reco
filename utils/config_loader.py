"""
配置加载模块 - 从 config.json 加载并合并配置
"""

import json
import platform
import os
import torch


def get_config(config_path="config.json"):
    """
    加载配置文件并根据当前平台返回合并后的配置字典。
    
    Args:
        config_path: 配置文件路径，默认为当前目录下的 config.json
        
    Returns:
        dict: 合并后的配置字典，包含 common + 平台特定配置 + device
    """
    # 加载 JSON 配置文件
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    # 从 common 获取基础配置
    merged_config = config_data.get("common", {}).copy()
    
    # 根据平台合并特定配置
    system = platform.system()
    if system == "Linux":
        platform_config = config_data.get("linux", {})
    elif system == "Windows":
        platform_config = config_data.get("windows", {})
    else:
        # 其他平台默认使用 Windows 配置
        platform_config = config_data.get("windows", {})
    
    merged_config.update(platform_config)
    
    # 自动检测并添加 device 配置
    merged_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return merged_config


def get_platform_name():
    """返回当前平台名称"""
    return platform.system()


def is_windows():
    """检查是否为 Windows 平台"""
    return platform.system() == "Windows"


def is_linux():
    """检查是否为 Linux 平台"""
    return platform.system() == "Linux"
