#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Dialog - 主入口脚本
基于PaddleOCR的对话框识别处理器
"""

import argparse
import sys

from ocr_dialog.core.processor import OCRDialogProcessor
from ocr_dialog.config.settings import AppConfig, OCRConfig, DialogMergerConfig, ProcessingConfig, LoggingConfig
from ocr_dialog.utils.logger import get_logger
from ocr_dialog.utils.image_processor import ImageProcessor


def create_config_from_args(args) -> AppConfig:
    """从命令行参数创建配置"""
    
    # OCR配置
    ocr_config = OCRConfig(
        language=args.lang,
        confidence_threshold=args.confidence,
        use_gpu=args.gpu
    )
    
    # 对话框合并配置
    dialog_config = DialogMergerConfig(
        max_distance=args.merge_distance,
        min_group_size=2
    )
    
    # 处理配置
    processing_config = ProcessingConfig(
        merge_dialogs=args.merge,
        enable_ocr=args.ocr,
        output_formats=["json", "image"] + (["sequence"] if args.ocr and args.merge else [])
    )
    
    # 日志配置
    logging_config = LoggingConfig(
        level=args.log_level,
        console_output=not args.quiet,
        file_path=args.log_file
    )
    
    return AppConfig(
        ocr=ocr_config,
        dialog_merger=dialog_config,
        processing=processing_config,
        logging=logging_config
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="OCR Dialog - 基于PaddleOCR的对话框识别处理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s image.jpg                          # 基本文本检测
  %(prog)s image.jpg --ocr                    # OCR文字识别
  %(prog)s image.jpg --ocr --merge            # OCR + 对话框合并
  %(prog)s image.jpg --ocr --merge --lang ch  # 指定简体中文
        """
    )
    
    # 基本参数
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument("--output", "-o", default="output", help="输出目录 (默认: output)")
    
    # OCR参数
    parser.add_argument("--ocr", action="store_true", help="启用OCR文字识别（默认仅检测）")
    parser.add_argument("--lang", "-l", default="chinese_cht", help="OCR识别语言 (默认: chinese_cht)")
    parser.add_argument("--confidence", "-c", type=float, default=0.75, help="置信度阈值 (默认: 0.75)")
    
    # 对话框合并参数
    parser.add_argument("--merge", "-m", action="store_true", help="合并可能的对话框区域")
    parser.add_argument("--merge-distance", type=float, default=40.0, help="对话框合并最大距离 (默认: 40.0)")
    
    # 系统参数
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")
    parser.add_argument("--log-file", help="日志文件路径")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，不输出到控制台")
    
    args = parser.parse_args()
    
    try:
        # 创建配置
        config = create_config_from_args(args)
        
        # 初始化处理器
        processor = OCRDialogProcessor(config)
        logger = get_logger(__name__, config.logging)
        
        # 验证输入文件  
        from pathlib import Path
        if not ImageProcessor.validate_image_file(args.image_path):
            logger.error(f"无效的图像文件: {args.image_path}")
            sys.exit(1)
        
        # 处理图像
        logger.info("开始处理...")
        result = processor.process_image(args.image_path, args.output)
        
        if result["status"] == "success":
            logger.info(f"处理完成！结果保存在目录: {args.output}")
            
            # 显示处理结果摘要
            for proc_result in result["results"]:
                logger.info(f"检测到 {proc_result['high_confidence_detections']} 个高置信度文本区域")
                
                if "merged_result" in proc_result:
                    merge_info = proc_result["merged_result"]
                    logger.info(f"生成 {merge_info['dialog_boxes']} 个对话框")
                    
                    if "total_dialogs" in merge_info:
                        logger.info(f"创建 {merge_info['total_dialogs']} 个按序对话")
        else:
            logger.warning("未检测到任何文本内容")
    
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理失败: {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()