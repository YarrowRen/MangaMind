"""
对话框可视化工具模块
"""

import json
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from .logger import get_logger

logger = get_logger(__name__)


class DialogVisualizer:
    """对话框可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        logger.debug("对话框可视化器初始化完成")
    
    @staticmethod
    def draw_text_boxes(image: np.ndarray, dt_polys: List[List[List[int]]], 
                       color=(0, 255, 0), thickness=2) -> np.ndarray:
        """绘制文本框"""
        for poly in dt_polys:
            points = np.array(poly, dtype=np.int32)
            cv2.polylines(image, [points], True, color, thickness)
        return image
    
    @staticmethod
    def draw_dialog_boxes(image: np.ndarray, dialog_boxes: List[Dict[str, Any]], 
                         merged_color=(255, 0, 0), single_color=(0, 255, 255), thickness=3) -> np.ndarray:
        """绘制对话框边界"""
        for i, box in enumerate(dialog_boxes):
            bbox = box['bbox']
            points = np.array(bbox, dtype=np.int32)
            
            # 根据是否为合并的群组选择颜色
            is_merged = box.get('is_merged', True)
            color = merged_color if is_merged else single_color
            
            cv2.polylines(image, [points], True, color, thickness)
            
            # 添加标签
            if is_merged:
                label = f"Dialog {i+1} ({box['text_count']} texts)"
            else:
                label = f"Text {i+1}"
            
            label_pos = (bbox[0][0], bbox[0][1] - 10)
            cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, color, 2)
        
        return image
    
    def visualize_merged_results(self, image_path: Union[str, Path], 
                               merged_json_path: Union[str, Path], 
                               output_path: Optional[Union[str, Path]] = None) -> str:
        """
        可视化合并结果
        
        Args:
            image_path: 原始图片路径
            merged_json_path: 合并结果JSON文件路径
            output_path: 输出图片路径
        
        Returns:
            输出图片路径
        """
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"无法读取图片: {image_path}")
            
            # 读取合并结果
            with open(merged_json_path, 'r', encoding='utf-8') as f:
                merged_result = json.load(f)
            
            # 创建副本用于绘制
            result_image = image.copy()
            
            # 绘制原始文本框（绿色）
            if 'original_dt_polys' in merged_result:
                result_image = self.draw_text_boxes(
                    result_image, 
                    merged_result['original_dt_polys'], 
                    color=(0, 255, 0), 
                    thickness=1
                )
            
            # 绘制合并后的对话框（红色=合并，黄色=单个）
            if 'dialog_boxes' in merged_result:
                result_image = self.draw_dialog_boxes(
                    result_image, 
                    merged_result['dialog_boxes'], 
                    merged_color=(0, 0, 255),  # 红色：合并的对话框
                    single_color=(0, 255, 255),  # 黄色：单个文本框
                    thickness=3
                )
            
            # 确定输出路径
            if output_path is None:
                base_name = Path(image_path).stem
                output_dir = Path(merged_json_path).parent
                output_path = output_dir / f"visualized_{base_name}.jpg"
            
            # 保存结果
            cv2.imwrite(str(output_path), result_image)
            logger.info(f"可视化图像已保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"可视化合并结果失败: {e}")
            raise
    
    def create_dialog_only_image(self, image_path: Union[str, Path], 
                                merged_json_path: Union[str, Path], 
                                output_path: Optional[Union[str, Path]] = None) -> str:
        """
        创建只显示对话框的图像
        
        Args:
            image_path: 原始图片路径
            merged_json_path: 合并结果JSON文件路径
            output_path: 输出图片路径
        
        Returns:
            输出图片路径
        """
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"无法读取图片: {image_path}")
            
            # 读取合并结果
            with open(merged_json_path, 'r', encoding='utf-8') as f:
                merged_result = json.load(f)
            
            # 创建副本用于绘制
            result_image = image.copy()
            
            # 只绘制合并后的对话框（红色=合并，黄色=单个）
            if 'dialog_boxes' in merged_result:
                result_image = self.draw_dialog_boxes(
                    result_image, 
                    merged_result['dialog_boxes'], 
                    merged_color=(0, 0, 255),  # 红色：合并的对话框
                    single_color=(0, 255, 255),  # 黄色：单个文本框
                    thickness=4
                )
            
            # 确定输出路径
            if output_path is None:
                base_name = Path(image_path).stem
                output_dir = Path(merged_json_path).parent
                output_path = output_dir / f"dialogs_only_{base_name}.jpg"
            
            # 保存结果
            cv2.imwrite(str(output_path), result_image)
            logger.info(f"对话框专用图像已保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"创建对话框专用图像失败: {e}")
            raise
    
    def create_comparison_image(self, image_path: Union[str, Path], 
                              merged_json_path: Union[str, Path], 
                              output_path: Optional[Union[str, Path]] = None) -> str:
        """
        创建对比图像（原图、检测结果、合并结果）
        
        Args:
            image_path: 原始图片路径
            merged_json_path: 合并结果JSON文件路径
            output_path: 输出图片路径
        
        Returns:
            输出图片路径
        """
        try:
            # 读取图片
            original_image = cv2.imread(str(image_path))
            if original_image is None:
                raise FileNotFoundError(f"无法读取图片: {image_path}")
            
            # 读取合并结果
            with open(merged_json_path, 'r', encoding='utf-8') as f:
                merged_result = json.load(f)
            
            # 创建三个版本的图像
            height, width = original_image.shape[:2]
            
            # 1. 原图
            image1 = original_image.copy()
            cv2.putText(image1, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            # 2. 检测结果
            image2 = original_image.copy()
            if 'original_dt_polys' in merged_result:
                image2 = self.draw_text_boxes(image2, merged_result['original_dt_polys'], 
                                            color=(0, 255, 0), thickness=2)
            cv2.putText(image2, "Text Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            # 3. 合并结果
            image3 = original_image.copy()
            if 'original_dt_polys' in merged_result:
                image3 = self.draw_text_boxes(image3, merged_result['original_dt_polys'], 
                                            color=(0, 255, 0), thickness=1)
            if 'dialog_boxes' in merged_result:
                image3 = self.draw_dialog_boxes(image3, merged_result['dialog_boxes'], 
                                              merged_color=(0, 0, 255), single_color=(0, 255, 255), thickness=3)
            cv2.putText(image3, "Dialog Merging", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
            
            # 调整图像大小以便并排显示
            target_width = 400
            scale = target_width / width
            target_height = int(height * scale)
            
            image1_resized = cv2.resize(image1, (target_width, target_height))
            image2_resized = cv2.resize(image2, (target_width, target_height))
            image3_resized = cv2.resize(image3, (target_width, target_height))
            
            # 水平拼接
            comparison_image = np.hstack([image1_resized, image2_resized, image3_resized])
            
            # 确定输出路径
            if output_path is None:
                base_name = Path(image_path).stem
                output_dir = Path(merged_json_path).parent
                output_path = output_dir / f"comparison_{base_name}.jpg"
            
            # 保存结果
            cv2.imwrite(str(output_path), comparison_image)
            logger.info(f"对比图像已保存至: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"创建对比图像失败: {e}")
            raise