"""
OCR Dialog - 基于PaddleOCR的对话框识别处理器

A professional OCR and dialog box merging tool for text recognition in images.
"""

__version__ = "1.0.0"
__author__ = "OCR Dialog Team"
__description__ = "Professional OCR and dialog box merging processor based on PaddleOCR"

from .core.ocr_engine import OCREngine
from .core.dialog_merger import DialogMerger
from .core.text_detector import TextDetector
from .core.processor import OCRDialogProcessor
from .utils.image_processor import ImageProcessor

__all__ = [
    "OCREngine",
    "DialogMerger", 
    "TextDetector",
    "OCRDialogProcessor",
    "ImageProcessor"
]