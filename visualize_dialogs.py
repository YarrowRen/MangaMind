#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话框可视化脚本
绘制检测到的文本区域和合并后的对话框
"""

import json
import cv2
import numpy as np
import argparse
import os
from typing import Dict, Any, List


def draw_text_boxes(image: np.ndarray, dt_polys: List[List[List[int]]], 
                   color=(0, 255, 0), thickness=2) -> np.ndarray:
    """绘制文本框"""
    for poly in dt_polys:
        points = np.array(poly, dtype=np.int32)
        cv2.polylines(image, [points], True, color, thickness)
    return image


def draw_dialog_boxes(image: np.ndarray, dialog_boxes: List[Dict[str, Any]], 
                     color=(255, 0, 0), thickness=3) -> np.ndarray:
    """绘制对话框边界"""
    for i, box in enumerate(dialog_boxes):
        bbox = box['bbox']
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(image, [points], True, color, thickness)
        
        # 添加标签
        label = f"Dialog {i+1} ({box['text_count']} texts)"
        label_pos = (bbox[0][0], bbox[0][1] - 10)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, color, 2)
    
    return image


def visualize_merged_results(image_path: str, merged_json_path: str, 
                           output_path: str = None) -> str:
    """
    可视化合并结果
    
    Args:
        image_path: 原始图片路径
        merged_json_path: 合并结果JSON文件路径
        output_path: 输出图片路径
    
    Returns:
        输出图片路径
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    
    # 读取合并结果
    with open(merged_json_path, 'r', encoding='utf-8') as f:
        merged_result = json.load(f)
    
    # 创建副本用于绘制
    result_image = image.copy()
    
    # 绘制原始文本框（绿色）
    if 'original_dt_polys' in merged_result:
        result_image = draw_text_boxes(
            result_image, 
            merged_result['original_dt_polys'], 
            color=(0, 255, 0), 
            thickness=1
        )
    
    # 绘制合并后的对话框（红色）
    if 'dialog_boxes' in merged_result:
        result_image = draw_dialog_boxes(
            result_image, 
            merged_result['dialog_boxes'], 
            color=(0, 0, 255), 
            thickness=3
        )
    
    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(merged_json_path)
        output_path = os.path.join(output_dir, f"visualized_{base_name}.jpg")
    
    # 保存结果
    cv2.imwrite(output_path, result_image)
    
    return output_path


def create_comparison_image(image_path: str, merged_json_path: str, 
                          output_path: str = None) -> str:
    """
    创建对比图像（原图、检测结果、合并结果）
    
    Args:
        image_path: 原始图片路径
        merged_json_path: 合并结果JSON文件路径
        output_path: 输出图片路径
    
    Returns:
        输出图片路径
    """
    # 读取图片
    original_image = cv2.imread(image_path)
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
        image2 = draw_text_boxes(image2, merged_result['original_dt_polys'], 
                                color=(0, 255, 0), thickness=2)
    cv2.putText(image2, "Text Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               1, (0, 255, 0), 2)
    
    # 3. 合并结果
    image3 = original_image.copy()
    if 'original_dt_polys' in merged_result:
        image3 = draw_text_boxes(image3, merged_result['original_dt_polys'], 
                                color=(0, 255, 0), thickness=1)
    if 'dialog_boxes' in merged_result:
        image3 = draw_dialog_boxes(image3, merged_result['dialog_boxes'], 
                                  color=(0, 0, 255), thickness=3)
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
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.dirname(merged_json_path)
        output_path = os.path.join(output_dir, f"comparison_{base_name}.jpg")
    
    # 保存结果
    cv2.imwrite(output_path, comparison_image)
    
    return output_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='对话框可视化脚本')
    parser.add_argument('image_path', help='原始图片路径')
    parser.add_argument('merged_json', help='合并结果JSON文件路径')
    parser.add_argument('--output', '-o', help='输出图片路径')
    parser.add_argument('--comparison', '-c', action='store_true', 
                       help='创建对比图像')
    
    args = parser.parse_args()
    
    try:
        if args.comparison:
            output_path = create_comparison_image(
                args.image_path, 
                args.merged_json, 
                args.output
            )
            print(f"对比图像已保存至: {output_path}")
        else:
            output_path = visualize_merged_results(
                args.image_path, 
                args.merged_json, 
                args.output
            )
            print(f"可视化图像已保存至: {output_path}")
            
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()