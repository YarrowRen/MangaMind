"""
对话框合并核心模块
"""

import math
from typing import List, Dict, Tuple, Any, Optional

from ..config.settings import DialogMergerConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Rectangular:
    """矩形类，用于碰撞检测"""
    
    def __init__(self, x: float, y: float, w: float, h: float):
        self.x0 = x
        self.y0 = y
        self.x1 = x + w
        self.y1 = y + h
        self.w = w
        self.h = h
    
    def __gt__(self, other):
        return self.w > other.w and self.h > other.h
    
    def __lt__(self, other):
        return self.w < other.w and self.h < other.h
    
    def collision(self, r2) -> bool:
        """检测与另一个矩形是否碰撞"""
        return (self.x0 < r2.x1 and self.y0 < r2.y1 and 
                self.x1 > r2.x0 and self.y1 > r2.y0)
    
    def distance_to(self, other) -> float:
        """计算到另一个矩形的距离"""
        center1_x = (self.x0 + self.x1) / 2
        center1_y = (self.y0 + self.y1) / 2
        center2_x = (other.x0 + other.x1) / 2
        center2_y = (other.y0 + other.y1) / 2
        
        return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def expand(self, expand_ratio: float = 1.5):
        """扩展矩形区域"""
        expand_w = self.w * expand_ratio - self.w
        expand_h = self.h * expand_ratio - self.h
        
        return Rectangular(
            self.x0 - expand_w / 2,
            self.y0 - expand_h / 2,
            self.w + expand_w,
            self.h + expand_h
        )


class DialogMerger:
    """对话框合并器"""
    
    def __init__(self, config: Optional[DialogMergerConfig] = None):
        """
        初始化对话框合并器
        
        Args:
            config: 对话框合并配置
        """
        self.config = config or DialogMergerConfig()
        logger.debug(f"对话框合并器初始化完成，配置: {self.config}")
    
    @staticmethod
    def poly_to_rect(poly_points: List[List[int]]) -> Rectangular:
        """将多边形坐标转换为矩形对象"""
        x_coords = [point[0] for point in poly_points]
        y_coords = [point[1] for point in poly_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return Rectangular(x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _find_nearby_texts(self, rect: Rectangular, all_rects: List[Tuple[Rectangular, int]], 
                          used_indices: set) -> List[int]:
        """查找附近的文本框（使用严格条件）"""
        nearby_indices = []
        
        # 使用配置的扩展比例
        expanded_rect = rect.expand(expand_ratio=self.config.expand_ratio)
        
        for other_rect, original_index in all_rects:
            if original_index in used_indices:
                continue
            
            # 优先考虑重叠
            if expanded_rect.collision(other_rect):
                nearby_indices.append(original_index)
            elif rect.distance_to(other_rect) <= self.config.max_distance:
                # 如果启用位置检查，进行额外的方向性检查
                if self.config.enable_position_check:
                    if self._check_position_compatibility(rect, other_rect):
                        nearby_indices.append(original_index)
                else:
                    nearby_indices.append(original_index)
        
        return nearby_indices
    
    def _check_position_compatibility(self, rect1: Rectangular, rect2: Rectangular) -> bool:
        """检查两个矩形是否在合理的相对位置"""
        rect1_center_x = (rect1.x0 + rect1.x1) / 2
        rect1_center_y = (rect1.y0 + rect1.y1) / 2
        rect2_center_x = (rect2.x0 + rect2.x1) / 2
        rect2_center_y = (rect2.y0 + rect2.y1) / 2
        
        h_distance = abs(rect1_center_x - rect2_center_x)
        v_distance = abs(rect1_center_y - rect2_center_y)
        
        # 检查是否在合理的相对位置
        return ((h_distance <= self.config.max_distance * 1.5 and v_distance <= rect1.h * 0.8) or
                (v_distance <= self.config.max_distance * 1.5 and h_distance <= rect1.w * 0.8))
    
    def merge_detection_result(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并检测结果中的对话框
        
        Args:
            detection_result: 检测结果
        
        Returns:
            合并后的结果
        """
        dt_polys = detection_result['dt_polys']
        dt_scores = detection_result['dt_scores']
        
        if not dt_polys:
            logger.warning("检测结果为空，无法进行合并")
            return detection_result
        
        logger.info(f"开始合并对话框，原始文本区域数量: {len(dt_polys)}")
        
        # 转换为矩形对象
        rectangles = []
        for i, poly in enumerate(dt_polys):
            rect = self.poly_to_rect(poly)
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
            self._find_connected_texts(rect, rectangles, used_indices, current_group)
            
            # 保留所有群组
            merged_groups.append(current_group)
        
        # 创建合并后的结果
        merged_result = self._create_merged_result(detection_result, merged_groups)
        
        dialog_count = len([g for g in merged_groups if len(g) >= self.config.min_group_size])
        single_count = len([g for g in merged_groups if len(g) == 1])
        
        logger.info(f"对话框合并完成，生成 {dialog_count} 个合并对话框，{single_count} 个单独文本框")
        
        return merged_result
    
    def merge_ocr_result(self, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并OCR结果中的对话框（支持文字排序）
        
        Args:
            ocr_result: OCR结果
        
        Returns:
            合并后的结果
        """
        dt_polys = ocr_result['dt_polys']
        dt_scores = ocr_result['dt_scores']
        dt_texts = ocr_result['dt_texts']
        
        if not dt_polys:
            logger.warning("OCR结果为空，无法进行合并")
            return ocr_result
        
        logger.info(f"开始合并OCR对话框，原始文本区域数量: {len(dt_polys)}")
        
        # 转换为矩形对象
        rectangles = []
        for i, poly in enumerate(dt_polys):
            rect = self.poly_to_rect(poly)
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
            self._find_connected_texts(rect, rectangles, used_indices, current_group)
            
            # 只保留群组大小满足要求的或单独的文本框
            if len(current_group) >= self.config.min_group_size or self.config.min_group_size == 1:
                merged_groups.append(current_group)
            else:
                # 对于小群组，将每个文本框作为单独的群组
                for idx in current_group:
                    merged_groups.append([idx])
        
        # 创建OCR合并结果
        merged_result = self._create_ocr_merged_result(ocr_result, merged_groups, dt_polys, dt_scores, dt_texts)
        
        dialog_count = len(merged_result.get("merged_groups", []))
        single_count = len(merged_result.get("single_texts", []))
        
        logger.info(f"OCR对话框合并完成，生成 {dialog_count} 个合并对话框，{single_count} 个单独文本框")
        
        return merged_result
    
    def _find_connected_texts(self, current_rect: Rectangular, rectangles: List[Tuple[Rectangular, int]], 
                            used_indices: set, current_group: List[int]):
        """递归查找相邻的文本框"""
        nearby_indices = self._find_nearby_texts(current_rect, rectangles, used_indices)
        
        for nearby_idx in nearby_indices:
            if nearby_idx not in used_indices:
                current_group.append(nearby_idx)
                used_indices.add(nearby_idx)
                
                # 递归查找与新加入文本框相邻的文本框
                nearby_rect = next(r for r, idx in rectangles if idx == nearby_idx)
                self._find_connected_texts(nearby_rect, rectangles, used_indices, current_group)
    
    def _create_merged_result(self, detection_result: Dict[str, Any], merged_groups: List[List[int]]) -> Dict[str, Any]:
        """创建标准合并结果"""
        dt_polys = detection_result['dt_polys']
        dt_scores = detection_result['dt_scores']
        
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
                'text_count': len(group_indices),
                'is_merged': len(group_indices) > 1
            }
            
            merged_result['dialog_boxes'].append(merged_box)
        
        # 按平均置信度排序
        merged_result['dialog_boxes'].sort(key=lambda x: x['avg_score'], reverse=True)
        
        return merged_result
    
    def _create_ocr_merged_result(self, ocr_result: Dict[str, Any], merged_groups: List[List[int]], 
                                dt_polys: List, dt_scores: List, dt_texts: List) -> Dict[str, Any]:
        """创建OCR合并结果"""
        merged_result = {
            "input_path": ocr_result["input_path"],
            "confidence_threshold": ocr_result["confidence_threshold"],
            "total_detections": ocr_result["total_detections"],
            "high_confidence_detections": ocr_result["high_confidence_detections"],
            "ocr_enabled": True,
            "original_dt_polys": dt_polys,
            "original_dt_scores": dt_scores,
            "original_dt_texts": dt_texts,
            "merged_groups": [],
            "single_texts": [],
            "dialog_boxes": []
        }
        
        for group_indices in merged_groups:
            if len(group_indices) >= self.config.min_group_size:
                # 对话框合并：按从右到左排序文字
                group_data = []
                for idx in group_indices:
                    poly = dt_polys[idx]
                    rect = self.poly_to_rect(poly)
                    center_x = (rect.x0 + rect.x1) / 2
                    group_data.append((idx, center_x, dt_texts[idx], dt_scores[idx], poly))
                
                # 按x坐标从右到左排序（x值大的在前）
                group_data.sort(key=lambda x: x[1], reverse=True)
                
                # 合并文字（用空格连接）
                merged_text = " ".join([item[2] for item in group_data])
                merged_score = sum([item[3] for item in group_data]) / len(group_data)
                
                # 计算合并后的边界框
                all_x = []
                all_y = []
                for _, _, _, _, poly in group_data:
                    for point in poly:
                        all_x.append(point[0])
                        all_y.append(point[1])
                
                merged_poly = [
                    [min(all_x), min(all_y)],
                    [max(all_x), min(all_y)],
                    [max(all_x), max(all_y)],
                    [min(all_x), max(all_y)]
                ]
                
                merged_result["merged_groups"].append({
                    "group_indices": group_indices,
                    "merged_text": merged_text,
                    "merged_score": merged_score,
                    "merged_poly": merged_poly,
                    "original_texts": [item[2] for item in group_data],
                    "original_scores": [item[3] for item in group_data],
                    "original_polys": [item[4] for item in group_data]
                })
                
                # 添加到dialog_boxes以兼容可视化函数
                merged_result["dialog_boxes"].append({
                    "indices": group_indices,
                    "bbox": merged_poly,
                    "individual_polys": [item[4] for item in group_data],
                    "individual_scores": [item[3] for item in group_data],
                    "avg_score": merged_score,
                    "text_count": len(group_indices),
                    "is_merged": True,
                    "merged_text": merged_text
                })
            else:
                # 单独的文本框
                for idx in group_indices:
                    merged_result["single_texts"].append({
                        "index": idx,
                        "text": dt_texts[idx],
                        "score": dt_scores[idx],
                        "poly": dt_polys[idx]
                    })
                    
                    # 添加到dialog_boxes以兼容可视化函数
                    merged_result["dialog_boxes"].append({
                        "indices": [idx],
                        "bbox": dt_polys[idx],
                        "individual_polys": [dt_polys[idx]],
                        "individual_scores": [dt_scores[idx]],
                        "avg_score": dt_scores[idx],
                        "text_count": 1,
                        "is_merged": False,
                        "merged_text": dt_texts[idx]
                    })
        
        return merged_result