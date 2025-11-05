"""
主处理器模块
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .text_detector import TextDetector
from .ocr_engine import OCREngine
from .dialog_merger import DialogMerger
from ..config.settings import AppConfig, get_config
from ..utils.logger import get_logger
from ..utils.image_processor import ImageProcessor, NumpyEncoder
from ..utils.visualizer import DialogVisualizer

logger = get_logger(__name__)


class DialogSequenceGenerator:
    """对话序列生成器"""
    
    @staticmethod
    def create_sequence(merged_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建按阅读顺序排列的对话序列
        按从右上角到左下角的顺序排列（以对话框右上顶点坐标为准）
        
        Args:
            merged_result: 合并后的OCR结果
        
        Returns:
            包含对话序列的结果
        """
        dialog_sequence = []
        all_dialogs = []
        
        # 添加合并的对话框
        for group in merged_result.get("merged_groups", []):
            merged_poly = group["merged_poly"]
            # 找到右上角坐标（x最大，y最小）
            x_coords = [point[0] for point in merged_poly]
            y_coords = [point[1] for point in merged_poly]
            right_top_x = max(x_coords)
            right_top_y = min(y_coords)
            
            all_dialogs.append({
                "type": "merged",
                "text": group["merged_text"],
                "score": group["merged_score"],
                "poly": merged_poly,
                "right_top_x": right_top_x,
                "right_top_y": right_top_y,
                "original_texts": group["original_texts"],
                "text_count": len(group["original_texts"])
            })
        
        # 添加单独的文本框
        for single in merged_result.get("single_texts", []):
            poly = single["poly"]
            # 找到右上角坐标
            x_coords = [point[0] for point in poly]
            y_coords = [point[1] for point in poly]
            right_top_x = max(x_coords)
            right_top_y = min(y_coords)
            
            all_dialogs.append({
                "type": "single",
                "text": single["text"],
                "score": single["score"],
                "poly": poly,
                "right_top_x": right_top_x,
                "right_top_y": right_top_y,
                "original_texts": [single["text"]],
                "text_count": 1
            })
        
        # 按从右上角到左下角排序
        # 主要按y坐标升序（从上到下），相同y坐标时按x坐标降序（从右到左）
        all_dialogs.sort(key=lambda d: (d["right_top_y"], -d["right_top_x"]))
        
        # 创建对话序列
        for i, dialog in enumerate(all_dialogs):
            dialog_sequence.append({
                "sequence_id": i + 1,
                "dialog_text": dialog["text"],
                "confidence_score": round(dialog["score"], 4),
                "dialog_type": dialog["type"],
                "text_count": dialog["text_count"],
                "position": {
                    "right_top_x": dialog["right_top_x"],
                    "right_top_y": dialog["right_top_y"]
                }
            })
        
        return {
            "input_path": merged_result["input_path"],
            "total_dialogs": len(dialog_sequence),
            "reading_order": "right_top_to_left_bottom",
            "dialog_sequence": dialog_sequence
        }


class OCRDialogProcessor:
    """OCR对话框处理器"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        初始化处理器
        
        Args:
            config: 应用配置，如果为None则使用默认配置
        """
        self.config = config or get_config()
        
        # 初始化日志
        self.logger = get_logger(__name__, self.config.logging)
        
        # 初始化组件
        self.text_detector = None
        self.ocr_engine = None
        self.dialog_merger = DialogMerger(self.config.dialog_merger)
        self.sequence_generator = DialogSequenceGenerator()
        self.visualizer = DialogVisualizer()
        
        self.logger.info("OCR对话框处理器初始化完成")
    
    def _get_text_detector(self) -> TextDetector:
        """获取文本检测器（懒加载）"""
        if self.text_detector is None:
            self.text_detector = TextDetector(self.config.ocr)
        return self.text_detector
    
    def _get_ocr_engine(self) -> OCREngine:
        """获取OCR引擎（懒加载）"""
        if self.ocr_engine is None:
            self.ocr_engine = OCREngine(self.config.ocr)
        return self.ocr_engine
    
    def process_image(self, image_path: Union[str, Path], output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        处理单个图像
        
        Args:
            image_path: 图像文件路径
            output_dir: 输出目录
        
        Returns:
            处理结果
        """
        image_path = Path(image_path)
        output_path = ImageProcessor.ensure_output_dir(output_dir)
        
        if not ImageProcessor.validate_image_file(image_path):
            raise ValueError(f"无效的图像文件: {image_path}")
        
        self.logger.info(f"开始处理图像: {image_path.name}")
        
        # 选择处理引擎
        if self.config.processing.enable_ocr:
            results = self._get_ocr_engine().recognize(image_path)
        else:
            results = self._get_text_detector().detect(image_path)
        
        if not results:
            self.logger.warning(f"图像 {image_path.name} 未检测到任何文本")
            return {"status": "no_text_detected", "input_path": str(image_path)}
        
        # 处理结果
        processed_results = []
        for result in results:
            processed_result = self._process_single_result(result, image_path, output_path)
            processed_results.append(processed_result)
        
        self.logger.info(f"图像 {image_path.name} 处理完成")
        return {
            "status": "success",
            "input_path": str(image_path),
            "results": processed_results
        }
    
    def _process_single_result(self, result: Dict[str, Any], image_path: Path, output_path: Path) -> Dict[str, Any]:
        """处理单个结果"""
        base_name = image_path.stem
        
        # 保存基础结果
        if result.get("ocr_enabled", False):
            result_file = output_path / f"ocr_{base_name}.json"
        else:
            result_file = output_path / f"detected_{base_name}.json"
        
        with open(result_file, "w", encoding="utf-8") as f:
            # 移除原始结果对象（不能序列化）
            save_result = {k: v for k, v in result.items() if k != "_raw_result"}
            json.dump(save_result, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
        
        self.logger.debug(f"基础结果已保存至: {result_file}")
        
        # 保存可视化图片
        if "_raw_result" in result:
            if result.get("ocr_enabled", False):
                img_file = output_path / f"ocr_{image_path.name}"
            else:
                img_file = output_path / f"detected_{image_path.name}"
            
            result["_raw_result"].save_to_img(save_path=str(img_file))
            self.logger.debug(f"可视化图片已保存至: {img_file}")
        
        processed_result = {
            "result_file": str(result_file),
            "high_confidence_detections": result["high_confidence_detections"]
        }
        
        # 对话框合并
        if self.config.processing.merge_dialogs:
            merged_result = self._process_dialog_merging(result, base_name, output_path)
            processed_result["merged_result"] = merged_result
        
        return processed_result
    
    def _process_dialog_merging(self, result: Dict[str, Any], base_name: str, output_path: Path) -> Dict[str, str]:
        """处理对话框合并"""
        self.logger.info("开始合并对话框...")
        
        if result.get("ocr_enabled", False):
            merged_result = self.dialog_merger.merge_ocr_result(result)
            merged_file = output_path / f"merged_ocr_{base_name}.json"
        else:
            merged_result = self.dialog_merger.merge_detection_result(result)
            merged_file = output_path / f"merged_{base_name}.json"
        
        # 保存合并结果
        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        self.logger.debug(f"合并结果已保存至: {merged_file}")
        
        # 生成可视化图像
        try:
            # 获取原始图像路径
            original_image_path = Path(result["input_path"])
            
            # 生成对话框专用图像
            dialog_img_path = self.visualizer.create_dialog_only_image(
                original_image_path, merged_file
            )
            
            # 生成完整可视化图像
            visual_img_path = self.visualizer.visualize_merged_results(
                original_image_path, merged_file
            )
            
            merge_info = {
                "merged_file": str(merged_file),
                "dialog_boxes": len(merged_result.get("dialog_boxes", [])),
                "dialog_image": dialog_img_path,
                "visualization_image": visual_img_path
            }
            
        except Exception as e:
            self.logger.warning(f"生成可视化图像失败: {e}")
            merge_info = {
                "merged_file": str(merged_file),
                "dialog_boxes": len(merged_result.get("dialog_boxes", [])),
            }
        
        # 生成对话序列（仅对OCR结果）
        if result.get("ocr_enabled", False) and "sequence" in self.config.processing.output_formats:
            sequence_result = self.sequence_generator.create_sequence(merged_result)
            sequence_file = output_path / f"dialog_sequence_{base_name}.json"
            
            with open(sequence_file, "w", encoding="utf-8") as f:
                json.dump(sequence_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            self.logger.debug(f"对话序列已保存至: {sequence_file}")
            merge_info["sequence_file"] = str(sequence_file)
            merge_info["total_dialogs"] = sequence_result["total_dialogs"]
        
        return merge_info