#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Dialog 批量处理脚本
支持按顺序读取指定文件夹下所有图像并进行OCR识别
生成简化的对话序列JSON文件
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from ocr_dialog.core.processor import OCRDialogProcessor
from ocr_dialog.config.settings import AppConfig, OCRConfig, DialogMergerConfig, ProcessingConfig, LoggingConfig
from ocr_dialog.utils.logger import get_logger
from ocr_dialog.utils.image_processor import ImageProcessor, NumpyEncoder


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, config: AppConfig):
        """
        初始化批量处理器
        
        Args:
            config: 应用配置
        """
        self.config = config
        self.logger = get_logger(__name__, config.logging)
        self.processor = OCRDialogProcessor(config)
    
    def process_single_image(self, image_path: Path, output_dir: Path, page_id: int) -> List[Dict[str, Any]]:
        """
        处理单个图像文件
        
        Args:
            image_path: 图像文件路径
            output_dir: 输出目录
            page_id: 页面ID
        
        Returns:
            简化的对话数据列表
        """
        self.logger.info(f"处理第 {page_id} 页: {image_path.name}")
        
        try:
            # 进行处理
            result = self.processor.process_image(image_path, output_dir)
            
            if result["status"] != "success":
                self.logger.warning(f"第 {page_id} 页未检测到文本")
                return []
            
            # 提取对话数据
            simplified_dialogs = []
            for proc_result in result["results"]:
                if "merged_result" in proc_result:
                    # 查找对话序列文件
                    sequence_file = proc_result["merged_result"].get("sequence_file")
                    if sequence_file and Path(sequence_file).exists():
                        with open(sequence_file, "r", encoding="utf-8") as f:
                            sequence_data = json.load(f)
                        
                        # 转换为简化格式
                        for dialog in sequence_data["dialog_sequence"]:
                            simplified_dialogs.append({
                                "page_id": page_id,
                                "sequence_id": dialog["sequence_id"],
                                "dialog_text": dialog["dialog_text"]
                            })
            
            self.logger.info(f"成功识别 {len(simplified_dialogs)} 个对话")
            return simplified_dialogs
            
        except Exception as e:
            self.logger.error(f"处理第 {page_id} 页时出错: {e}")
            return []
    
    def batch_process_images(self, input_folder: str, output_dir: str) -> None:
        """
        批量处理图像文件
        
        Args:
            input_folder: 输入图像文件夹
            output_dir: 输出目录
        """
        # 创建输出目录
        output_path = ImageProcessor.ensure_output_dir(output_dir)
        
        # 获取所有图像文件
        self.logger.info(f"扫描文件夹: {input_folder}")
        image_files = ImageProcessor.get_image_files(input_folder)
        
        if not image_files:
            self.logger.warning("未找到任何图像文件")
            return
        
        self.logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 存储所有对话数据
        all_dialogs = []
        
        # 处理每个图像
        for page_id, image_path in enumerate(image_files, 1):
            dialogs = self.process_single_image(image_path, output_path, page_id)
            all_dialogs.extend(dialogs)
        
        # 保存简化的批处理结果
        batch_result = {
            "total_pages": len(image_files),
            "total_dialogs": len(all_dialogs),
            "processing_config": {
                "confidence_threshold": self.config.ocr.confidence_threshold,
                "ocr_language": self.config.ocr.language,
                "merge_dialogs": self.config.processing.merge_dialogs,
                "enable_ocr": self.config.processing.enable_ocr
            },
            "dialogs": all_dialogs
        }
        
        # 保存到批处理结果文件
        batch_json_path = output_path / "batch_dialogs.json"
        with open(batch_json_path, "w", encoding="utf-8") as f:
            json.dump(batch_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        self.logger.info("批处理完成!")
        self.logger.info(f"总页数: {len(image_files)}")
        self.logger.info(f"总对话数: {len(all_dialogs)}")
        self.logger.info(f"批处理结果已保存至: {batch_json_path}")


def create_batch_config(args) -> AppConfig:
    """从命令行参数创建批处理配置"""
    
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
    
    # 处理配置（批处理模式下启用OCR和对话框合并）
    processing_config = ProcessingConfig(
        merge_dialogs=True,
        enable_ocr=True,
        output_formats=["json", "image", "sequence"]
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
        description="OCR Dialog批量处理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s input_folder                    # 使用默认设置批量处理
  %(prog)s input_folder --lang ch         # 指定简体中文
  %(prog)s input_folder --confidence 0.8  # 调整置信度阈值
        """
    )
    
    # 基本参数
    parser.add_argument("input_folder", help="输入图像文件夹路径")
    parser.add_argument("--output", "-o", default="batch_output", help="输出目录 (默认: batch_output)")
    
    # OCR参数
    parser.add_argument("--lang", "-l", default="chinese_cht", help="OCR识别语言 (默认: chinese_cht)")
    parser.add_argument("--confidence", "-c", type=float, default=0.75, help="置信度阈值 (默认: 0.75)")
    
    # 对话框合并参数
    parser.add_argument("--merge-distance", type=float, default=40.0, help="对话框合并最大距离 (默认: 40.0)")
    
    # 系统参数
    parser.add_argument("--gpu", action="store_true", help="使用GPU加速")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="日志级别")
    parser.add_argument("--log-file", help="日志文件路径")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，不输出到控制台")
    
    args = parser.parse_args()
    
    try:
        # 创建配置
        config = create_batch_config(args)
        
        # 验证输入文件夹
        if not Path(args.input_folder).exists():
            print(f"错误: 输入文件夹不存在: {args.input_folder}", file=sys.stderr)
            sys.exit(1)
        
        # 创建批处理器并运行
        batch_processor = BatchProcessor(config)
        batch_processor.batch_process_images(args.input_folder, args.output)
        
    except KeyboardInterrupt:
        print("用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"批处理失败: {e}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback
            print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()