#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话框文本合并脚本
基于矩形碰撞检测，将可能属于同一对话框的文本区域进行合并
"""

import json
import math
from typing import List, Dict, Tuple, Any


class Rectangular:
    """矩形类，用于碰撞检测"""
    
    def __init__(self, x, y, w, h):
        self.x0 = x
        self.y0 = y
        self.x1 = x + w
        self.y1 = y + h
        self.w = w
        self.h = h
    
    def __gt__(self, other):
        if self.w > other.w and self.h > other.h:
            return True
        return False
    
    def __lt__(self, other):
        if self.w < other.w and self.h < other.h:
            return True
        return False
    
    def collision(self, r2):
        """检测与另一个矩形是否碰撞"""
        if self.x0 < r2.x1 and self.y0 < r2.y1 and self.x1 > r2.x0 and self.y1 > r2.y0:
            return True
        return False
    
    def distance_to(self, other):
        """计算到另一个矩形的距离"""
        # 计算两个矩形中心点的距离
        center1_x = (self.x0 + self.x1) / 2
        center1_y = (self.y0 + self.y1) / 2
        center2_x = (other.x0 + other.x1) / 2
        center2_y = (other.y0 + other.y1) / 2
        
        return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def expand(self, expand_ratio=1.5):
        """扩展矩形区域"""
        expand_w = self.w * expand_ratio - self.w
        expand_h = self.h * expand_ratio - self.h
        
        return Rectangular(
            self.x0 - expand_w / 2,
            self.y0 - expand_h / 2,
            self.w + expand_w,
            self.h + expand_h
        )


def poly_to_rect(poly_points: List[List[int]]) -> Rectangular:
    """将多边形坐标转换为矩形对象"""
    # 获取四个顶点的坐标
    x_coords = [point[0] for point in poly_points]
    y_coords = [point[1] for point in poly_points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return Rectangular(x_min, y_min, x_max - x_min, y_max - y_min)


def find_nearby_texts(rect: Rectangular, all_rects: List[Tuple[Rectangular, int]], 
                      used_indices: set, max_distance: float = 100) -> List[int]:
    """查找附近的文本框"""
    nearby_indices = []
    
    # 扩展当前矩形以增加检测范围
    expanded_rect = rect.expand(expand_ratio=2.0)
    
    for other_rect, original_index in all_rects:
        if original_index in used_indices:
            continue
            
        # 检查是否在扩展的矩形范围内或距离足够近
        if (expanded_rect.collision(other_rect) or 
            rect.distance_to(other_rect) <= max_distance):
            nearby_indices.append(original_index)
    
    return nearby_indices


def merge_dialog_boxes(detection_result: Dict[str, Any], 
                       max_distance: float = 150,
                       min_group_size: int = 2) -> Dict[str, Any]:
    """
    合并对话框文本区域
    
    Args:
        detection_result: 文本检测结果
        max_distance: 最大合并距离
        min_group_size: 最小群组大小
    
    Returns:
        合并后的结果
    """
    dt_polys = detection_result['dt_polys']
    dt_scores = detection_result['dt_scores']
    
    if not dt_polys:
        return detection_result
    
    # 转换为矩形对象
    rectangles = []
    for i, poly in enumerate(dt_polys):
        rect = poly_to_rect(poly)
        rectangles.append((rect, i))
    
    # 按面积排序，从大到小处理
    rectangles.sort(key=lambda x: x[0].w * x[0].h, reverse=True)
    
    # 合并逻辑
    merged_groups = []
    used_indices = set()
    
    for rect, original_index in rectangles:
        if original_index in used_indices:
            continue
        
        # 创建新的群组
        current_group = [original_index]
        used_indices.add(original_index)
        
        # 递归查找相邻的文本框
        def find_connected_texts(current_rect, current_indices):
            nearby_indices = find_nearby_texts(current_rect, rectangles, used_indices, max_distance)
            
            for nearby_idx in nearby_indices:
                if nearby_idx not in used_indices:
                    current_indices.append(nearby_idx)
                    used_indices.add(nearby_idx)
                    
                    # 递归查找与新加入文本框相邻的文本框
                    nearby_rect = next(r for r, idx in rectangles if idx == nearby_idx)
                    find_connected_texts(nearby_rect, current_indices)
        
        find_connected_texts(rect, current_group)
        
        # 只保留包含足够文本框的群组
        if len(current_group) >= min_group_size:
            merged_groups.append(current_group)
    
    # 创建合并后的结果
    merged_result = {
        'input_path': detection_result['input_path'],
        'confidence_threshold': detection_result['confidence_threshold'],
        'total_detections': detection_result['total_detections'],
        'high_confidence_detections': detection_result['high_confidence_detections'],
        'original_dt_polys': dt_polys,
        'original_dt_scores': dt_scores,
        'merged_groups': merged_groups,
        'dialog_boxes': []
    }
    
    # 为每个合并的群组创建边界框
    for group_indices in merged_groups:
        group_polys = [dt_polys[i] for i in group_indices]
        group_scores = [dt_scores[i] for i in group_indices]
        
        # 计算合并边界框
        all_x = []
        all_y = []
        for poly in group_polys:
            for point in poly:
                all_x.append(point[0])
                all_y.append(point[1])
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        merged_box = {
            'indices': group_indices,
            'bbox': [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
            'individual_polys': group_polys,
            'individual_scores': group_scores,
            'avg_score': sum(group_scores) / len(group_scores),
            'text_count': len(group_indices)
        }
        
        merged_result['dialog_boxes'].append(merged_box)
    
    # 按平均置信度排序
    merged_result['dialog_boxes'].sort(key=lambda x: x['avg_score'], reverse=True)
    
    return merged_result


def visualize_merged_boxes(merged_result: Dict[str, Any]) -> str:
    """生成可视化信息"""
    dialog_boxes = merged_result['dialog_boxes']
    
    if not dialog_boxes:
        return "未检测到可合并的对话框区域"
    
    info = f"检测到 {len(dialog_boxes)} 个对话框区域:\n\n"
    
    for i, box in enumerate(dialog_boxes, 1):
        bbox = box['bbox']
        width = bbox[1][0] - bbox[0][0]
        height = bbox[2][1] - bbox[0][1]
        
        info += f"对话框 {i}:\n"
        info += f"  - 包含文本框数量: {box['text_count']}\n"
        info += f"  - 平均置信度: {box['avg_score']:.3f}\n"
        info += f"  - 边界框大小: {width} x {height}\n"
        info += f"  - 位置: ({bbox[0][0]}, {bbox[0][1]}) 到 ({bbox[2][0]}, {bbox[2][1]})\n"
        info += f"  - 包含的文本框索引: {box['indices']}\n\n"
    
    return info


def process_detection_file(json_file_path: str, output_file_path: str = None,
                         max_distance: float = 150, min_group_size: int = 2) -> Dict[str, Any]:
    """
    处理检测结果文件
    
    Args:
        json_file_path: 输入的JSON文件路径
        output_file_path: 输出文件路径
        max_distance: 最大合并距离
        min_group_size: 最小群组大小
    
    Returns:
        合并后的结果
    """
    # 读取检测结果
    with open(json_file_path, 'r', encoding='utf-8') as f:
        detection_result = json.load(f)
    
    # 执行合并
    merged_result = merge_dialog_boxes(detection_result, max_distance, min_group_size)
    
    # 保存结果
    if output_file_path is None:
        output_file_path = json_file_path.replace('.json', '_merged.json')
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)
    
    return merged_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='对话框文本合并脚本')
    parser.add_argument('json_file', help='输入的检测结果JSON文件')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--distance', '-d', type=float, default=150, 
                       help='最大合并距离 (默认: 150)')
    parser.add_argument('--min-size', '-m', type=int, default=2,
                       help='最小群组大小 (默认: 2)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细信息')
    
    args = parser.parse_args()
    
    try:
        merged_result = process_detection_file(
            args.json_file, 
            args.output, 
            args.distance, 
            args.min_size
        )
        
        if args.verbose:
            print(visualize_merged_boxes(merged_result))
        
        print(f"对话框合并完成！")
        output_path = args.output or args.json_file.replace('.json', '_merged.json')
        print(f"结果保存至: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")