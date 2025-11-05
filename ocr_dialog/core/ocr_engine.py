"""
OCR引擎核心模块
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from paddleocr import PaddleOCR
from ..config.settings import OCRConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OCREngine:
    """OCR识别引擎"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        初始化OCR引擎
        
        Args:
            config: OCR配置，如果为None则使用默认配置
        """
        self.config = config or OCRConfig()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化OCR模型"""
        try:
            logger.info(f"正在初始化OCR模型（语言: {self.config.language}）...")
            self.model = PaddleOCR(
                lang=self.config.language,
                text_det_limit_type=self.config.det_limit_type,
                text_det_limit_side_len=self.config.det_limit_side_len,
                use_doc_orientation_classify=self.config.use_doc_orientation_classify,
                use_doc_unwarping=self.config.use_doc_unwarping,
            )
            logger.info("OCR模型初始化完成")
        except Exception as e:
            logger.error(f"OCR模型初始化失败: {e}")
            raise
    
    def recognize(self, image_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        识别图像中的文本
        
        Args:
            image_path: 图像文件路径
        
        Returns:
            识别结果列表
        """
        if not self.model:
            raise RuntimeError("OCR模型未初始化")
        
        try:
            logger.debug(f"开始OCR识别: {image_path}")
            ocr_results = self.model.predict(str(image_path))
            
            processed_results = []
            for res in ocr_results:
                result_data = res.json
                
                if "res" in result_data:
                    input_path = result_data["res"]["input_path"]
                    res_data = result_data["res"]
                    
                    # 检查OCR结果结构
                    if all(key in res_data for key in ["rec_texts", "rec_scores", "rec_polys"]):
                        rec_texts = res_data["rec_texts"]
                        rec_scores = res_data["rec_scores"]
                        rec_polys = res_data["rec_polys"]
                        
                        # 过滤置信度
                        dt_polys = []
                        dt_scores = []
                        dt_texts = []
                        
                        for text, score, poly in zip(rec_texts, rec_scores, rec_polys):
                            if score > self.config.confidence_threshold:
                                dt_polys.append(poly)
                                dt_scores.append(score)
                                dt_texts.append(text)
                        
                        if dt_polys:
                            filtered_result = {
                                "input_path": input_path,
                                "confidence_threshold": self.config.confidence_threshold,
                                "total_detections": len(rec_texts),
                                "high_confidence_detections": len(dt_polys),
                                "dt_polys": dt_polys,
                                "dt_scores": dt_scores,
                                "dt_texts": dt_texts,
                                "ocr_enabled": True,
                                "_raw_result": res  # 保存原始结果
                            }
                            
                            processed_results.append(filtered_result)
                            
                            logger.info(
                                f"OCR识别到 {len(rec_texts)} 个文本区域，"
                                f"其中 {len(dt_polys)} 个置信度高于 {self.config.confidence_threshold}"
                            )
                        else:
                            logger.warning(f"未检测到置信度高于 {self.config.confidence_threshold} 的文本区域")
                    else:
                        logger.error("OCR结果格式不符合预期")
                else:
                    logger.error("OCR结果格式错误")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            raise