"""
图像处理工具模块
"""

import os
import re
from pathlib import Path
from typing import List, Union, Optional
import json
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


class ImageProcessor:
    """图像处理工具类"""
    
    # 支持的图像格式
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    @classmethod
    def get_image_files(cls, folder_path: Union[str, Path], recursive: bool = False) -> List[Path]:
        """
        获取文件夹中的所有图像文件，按文件名自然排序
        
        Args:
            folder_path: 图像文件夹路径
            recursive: 是否递归搜索子目录
        
        Returns:
            排序后的图像文件路径列表
        """
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"文件夹不存在或不是目录: {folder_path}")
        
        # 获取所有图像文件
        image_files = []
        
        if recursive:
            for file_path in folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_FORMATS:
                    image_files.append(file_path)
        else:
            for file_path in folder.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_FORMATS:
                    image_files.append(file_path)
        
        # 自然排序
        image_files.sort(key=cls._natural_sort_key)
        
        logger.info(f"在 {folder_path} 中找到 {len(image_files)} 个图像文件")
        return image_files
    
    @staticmethod
    def _natural_sort_key(path: Path) -> List:
        """自然排序关键字函数"""
        parts = re.split(r'(\d+)', path.name)
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    @staticmethod
    def validate_image_file(file_path: Union[str, Path]) -> bool:
        """
        验证图像文件是否有效
        
        Args:
            file_path: 图像文件路径
        
        Returns:
            文件是否有效
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"图像文件不存在: {file_path}")
            return False
        
        if not path.is_file():
            logger.error(f"路径不是文件: {file_path}")
            return False
        
        if path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
            logger.error(f"不支持的图像格式: {path.suffix}")
            return False
        
        return True
    
    @staticmethod
    def ensure_output_dir(output_dir: Union[str, Path]) -> Path:
        """
        确保输出目录存在
        
        Args:
            output_dir: 输出目录路径
        
        Returns:
            输出目录Path对象
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"输出目录已准备: {output_path}")
        return output_path


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数组"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)