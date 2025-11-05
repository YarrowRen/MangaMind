"""
配置管理模块
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class OCRConfig:
    """OCR配置"""
    language: str = "chinese_cht"
    confidence_threshold: float = 0.75
    det_limit_type: str = "max"
    det_limit_side_len: int = 960
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_gpu: bool = False
    device: str = "cpu"


@dataclass  
class DialogMergerConfig:
    """对话框合并配置"""
    max_distance: float = 40.0
    min_group_size: int = 2
    expand_ratio: float = 1.3
    enable_position_check: bool = True


@dataclass
class ProcessingConfig:
    """处理配置"""
    merge_dialogs: bool = True
    enable_ocr: bool = True
    output_formats: list = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["json", "sequence", "image"]


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    console_output: bool = True
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """应用程序主配置"""
    ocr: OCRConfig = None
    dialog_merger: DialogMergerConfig = None
    processing: ProcessingConfig = None
    logging: LoggingConfig = None
    
    def __post_init__(self):
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.dialog_merger is None:
            self.dialog_merger = DialogMergerConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.logging is None:
            self.logging = LoggingConfig()


# 默认配置实例
DEFAULT_CONFIG = AppConfig()


def load_config_from_file(config_path: str) -> AppConfig:
    """从文件加载配置"""
    # TODO: 实现从YAML/JSON文件加载配置
    pass


def get_config() -> AppConfig:
    """获取当前配置"""
    config_file = os.getenv("MANGA_MIND_CONFIG")
    if config_file and Path(config_file).exists():
        return load_config_from_file(config_file)
    return DEFAULT_CONFIG