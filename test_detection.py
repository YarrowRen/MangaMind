#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本 - 下载测试图片并运行文本检测
"""

import os
import sys
import urllib.request
from detect_traditional_chinese import detect_text_with_confidence


def download_test_image():
    """下载测试图片"""
    test_url = "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png"
    test_image_path = "test_image.png"
    
    if not os.path.exists(test_image_path):
        print("正在下载测试图片...")
        try:
            urllib.request.urlretrieve(test_url, test_image_path)
            print(f"测试图片已下载: {test_image_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            return None
    else:
        print(f"使用现有测试图片: {test_image_path}")
    
    return test_image_path


def main():
    """主测试函数"""
    print("=== PaddleOCR 文本检测测试 ===\n")
    
    # 下载测试图片
    test_image = download_test_image()
    if not test_image:
        print("无法获取测试图片")
        return
    
    # 执行文本检测测试
    try:
        print("\n开始文本检测测试...")
        results = detect_text_with_confidence(
            image_path=test_image,
            confidence_threshold=0.7,  # 使用较低阈值以确保能检测到结果
            output_dir="test_output"
        )
        
        if results:
            print(f"\n✅ 测试成功！")
            print(f"检测结果保存在: test_output/ 目录")
            print(f"您可以查看可视化图片和JSON结果文件")
        else:
            print("⚠️ 未检测到符合条件的文本区域，可能需要调整置信度阈值")
            
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install paddleocr")
    except Exception as e:
        print(f"❌ 测试失败: {e}")


if __name__ == "__main__":
    main()