import json
import os
import platform

import torch


def get_default_config_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base_dir, path_value))


def get_config(config_path=None):
    if config_path is None:
        config_path = get_default_config_path()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    config = config_data.get("common", {}).copy()

    system = platform.system()
    if system == "Linux":
        platform_config = config_data.get("linux", {})
    elif system == "Windows":
        platform_config = config_data.get("windows", {})
    else:
        platform_config = config_data.get("windows", {})

    config.update(platform_config)

    base_dir = os.path.dirname(os.path.abspath(config_path))
    config["data_dir"] = _resolve_path(base_dir, config.get("data_dir", "../dataset"))
    config["checkpoint_dir"] = _resolve_path(
        base_dir, config.get("checkpoint_dir", "../checkpoint/split_ultralight_parallel_me_gru")
    )

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def get_platform_name():
    return platform.system()
