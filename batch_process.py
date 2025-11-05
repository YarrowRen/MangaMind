#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量图像文本识别脚本
支持按顺序读取指定文件夹下所有图像并进行OCR识别
生成简化的对话序列JSON文件
"""

import argparse
import json
import os
import sys
from pathlib import Path
import re
from typing import List, Dict, Any

from text_detection import detect_text_with_confidence, create_dialog_sequence, NumpyEncoder


def get_image_files(folder_path: str) -> List[Path]:
    """
    获取文件夹中的所有图像文件，按文件名自然排序
    
    Args:
        folder_path: 图像文件夹路径
    
    Returns:
        排序后的图像文件路径列表
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"文件夹不存在或不是目录: {folder_path}")
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    # 获取所有图像文件
    image_files = []
    for file_path in folder.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    # 自然排序（处理数字序列）
    def natural_sort_key(path):
        """自然排序关键字函数"""
        parts = re.split(r'(\d+)', path.name)
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    image_files.sort(key=natural_sort_key)
    return image_files


def process_single_image(image_path: Path, output_dir: str, page_id: int, 
                        confidence_threshold: float, ocr_lang: str) -> List[Dict[str, Any]]:
    """
    处理单个图像文件
    
    Args:
        image_path: 图像文件路径
        output_dir: 输出目录
        page_id: 页面ID
        confidence_threshold: 置信度阈值
        ocr_lang: OCR语言
    
    Returns:
        简化的对话数据列表
    """
    print(f"处理第 {page_id} 页: {image_path.name}")
    
    try:
        # 进行OCR识别和对话框合并
        results = detect_text_with_confidence(
            image_path=str(image_path),
            confidence_threshold=confidence_threshold,
            output_dir=output_dir,
            merge_dialogs=True,
            enable_ocr=True,
            ocr_lang=ocr_lang
        )
        
        if not results:
            print(f"  警告: 第 {page_id} 页未检测到文本")
            return []
        
        # 获取合并结果（假设只有一个结果）
        filtered_result = results[0]
        if 'merged_result' not in filtered_result:
            print(f"  警告: 第 {page_id} 页未找到合并结果")
            return []
        
        # 生成对话序列
        merged_result = filtered_result['merged_result']
        dialog_sequence_data = create_dialog_sequence(merged_result)
        
        # 转换为简化格式
        simplified_dialogs = []
        for dialog in dialog_sequence_data['dialog_sequence']:
            simplified_dialogs.append({
                "page_id": page_id,
                "sequence_id": dialog['sequence_id'],
                "dialog_text": dialog['dialog_text']
            })
        
        print(f"  成功识别 {len(simplified_dialogs)} 个对话")
        return simplified_dialogs
        
    except Exception as e:
        print(f"  错误: 处理第 {page_id} 页时出错: {e}")
        return []


def batch_process_images(input_folder: str, output_dir: str, confidence_threshold: float = 0.75, 
                        ocr_lang: str = "chinese_cht") -> None:
    """
    批量处理图像文件
    
    Args:
        input_folder: 输入图像文件夹
        output_dir: 输出目录
        confidence_threshold: 置信度阈值
        ocr_lang: OCR识别语言
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    print(f"扫描文件夹: {input_folder}")
    image_files = get_image_files(input_folder)
    
    if not image_files:
        print("未找到任何图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 存储所有对话数据
    all_dialogs = []
    
    # 处理每个图像
    for page_id, image_path in enumerate(image_files, 1):
        dialogs = process_single_image(
            image_path, output_dir, page_id, confidence_threshold, ocr_lang
        )
        all_dialogs.extend(dialogs)
    
    # 保存简化的批处理结果
    batch_result = {
        "total_pages": len(image_files),
        "total_dialogs": len(all_dialogs),
        "processing_config": {
            "confidence_threshold": confidence_threshold,
            "ocr_language": ocr_lang,
            "merge_dialogs": True
        },
        "dialogs": all_dialogs
    }
    
    # 保存到批处理结果文件
    batch_json_path = os.path.join(output_dir, "batch_dialogs.json")
    with open(batch_json_path, "w", encoding="utf-8") as f:
        json.dump(batch_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    print(f"\n批处理完成!")
    print(f"总页数: {len(image_files)}")
    print(f"总对话数: {len(all_dialogs)}")
    print(f"批处理结果已保存至: {batch_json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量图像文本识别脚本")
    parser.add_argument("input_folder", help="输入图像文件夹路径")
    parser.add_argument("--output", "-o", default="batch_output", help="输出目录 (默认: batch_output)")
    parser.add_argument("--confidence", "-c", type=float, default=0.75, help="置信度阈值 (默认: 0.75)")
    parser.add_argument("--lang", "-l", default="chinese_cht", help="OCR识别语言 (默认: chinese_cht)")
    
    args = parser.parse_args()
    
    try:
        batch_process_images(
            input_folder=args.input_folder,
            output_dir=args.output,
            confidence_threshold=args.confidence,
            ocr_lang=args.lang
        )
    except Exception as e:
        print(f"批处理失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()