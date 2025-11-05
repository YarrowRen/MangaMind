"""
文本检测核心模块
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from paddleocr import TextDetection
from ..config.settings import OCRConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TextDetector:
    """文本检测器"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        初始化文本检测器
        
        Args:
            config: OCR配置，如果为None则使用默认配置
        """
        self.config = config or OCRConfig()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化检测模型"""
        try:
            logger.info("正在初始化文本检测模型...")
            self.model = TextDetection(
                model_name="PP-OCRv5_server_det",
                device=self.config.device,
                box_thresh=self.config.confidence_threshold,
            )
            logger.info("文本检测模型初始化完成")
        except Exception as e:
            logger.error(f"文本检测模型初始化失败: {e}")
            raise
    
    def detect(self, image_path: Union[str, Path], batch_size: int = 1) -> List[Dict[str, Any]]:
        """
        检测图像中的文本区域
        
        Args:
            image_path: 图像文件路径
            batch_size: 批处理大小
        
        Returns:
            检测结果列表
        """
        if not self.model:
            raise RuntimeError("文本检测模型未初始化")
        
        try:
            logger.debug(f"开始检测图像: {image_path}")
            results = self.model.predict(str(image_path), batch_size=batch_size)
            
            processed_results = []
            for res in results:
                result_data = res.json
                input_path = result_data["res"]["input_path"]
                dt_polys = result_data["res"]["dt_polys"]
                dt_scores = result_data["res"]["dt_scores"]
                
                # 过滤置信度
                high_confidence_indices = [
                    i for i, score in enumerate(dt_scores) 
                    if score > self.config.confidence_threshold
                ]
                
                if high_confidence_indices:
                    filtered_polys = [dt_polys[i] for i in high_confidence_indices]
                    filtered_scores = [dt_scores[i] for i in high_confidence_indices]
                    
                    filtered_result = {
                        "input_path": input_path,
                        "confidence_threshold": self.config.confidence_threshold,
                        "total_detections": len(dt_scores),
                        "high_confidence_detections": len(high_confidence_indices),
                        "dt_polys": filtered_polys,
                        "dt_scores": filtered_scores,
                        "ocr_enabled": False,
                        "_raw_result": res  # 保存原始结果用于后续处理
                    }
                    
                    processed_results.append(filtered_result)
                    
                    logger.info(
                        f"检测到 {len(dt_scores)} 个文本区域，"
                        f"其中 {len(high_confidence_indices)} 个置信度高于 {self.config.confidence_threshold}"
                    )
                else:
                    logger.warning(f"未检测到置信度高于 {self.config.confidence_threshold} 的文本区域")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"文本检测失败: {e}")
            raise