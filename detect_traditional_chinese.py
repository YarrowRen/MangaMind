#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
繁体中文文本检测脚本
使用PaddleOCR检测图片中的文本区域，过滤置信度高于0.75的结果
"""

import argparse
import json
import os
import sys

import numpy as np
from paddleocr import TextDetection
from dialog_merger import merge_dialog_boxes, visualize_merged_boxes
from visualize_dialogs import create_dialog_only_image


def detect_text_with_confidence(image_path, confidence_threshold=0.75, output_dir="output", merge_dialogs=False):
    """
    检测图片中的文本区域，过滤高置信度结果

    Args:
        image_path (str): 输入图片路径
        confidence_threshold (float): 置信度阈值，默认0.75
        output_dir (str): 输出目录
        merge_dialogs (bool): 是否合并对话框，默认False

    Returns:
        dict: 检测结果
    """
    # 检查输入文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图片不存在: {image_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 初始化文本检测模型（使用服务端高精度模型）
    print("正在初始化文本检测模型...")
    model = TextDetection(
        model_name="PP-OCRv5_server_det",
        device="cpu",  # 可根据需要改为"gpu"
        box_thresh=confidence_threshold,  # 设置检测框置信度阈值
    )

    # 进行文本检测
    print(f"正在检测图片: {image_path}")
    results = model.predict(image_path, batch_size=1)

    # 处理检测结果
    filtered_results = []
    for res in results:
        # 直接访问结果属性
        result_data = res.json
        input_path = result_data["res"]["input_path"]
        dt_polys = result_data["res"]["dt_polys"]
        dt_scores = result_data["res"]["dt_scores"]

        # 过滤置信度大于阈值的结果
        high_confidence_indices = [i for i, score in enumerate(dt_scores) if score > confidence_threshold]

        if high_confidence_indices:
            filtered_polys = [dt_polys[i] for i in high_confidence_indices]
            filtered_scores = [dt_scores[i] for i in high_confidence_indices]

            filtered_result = {
                "input_path": input_path,
                "confidence_threshold": confidence_threshold,
                "total_detections": len(dt_scores),
                "high_confidence_detections": len(high_confidence_indices),
                "dt_polys": filtered_polys,
                "dt_scores": filtered_scores,
            }

            filtered_results.append(filtered_result)

            print(
                f"检测到 {len(dt_scores)} 个文本区域，其中 {len(high_confidence_indices)} 个置信度高于 {confidence_threshold}"
            )

            # 保存可视化图片
            img_output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
            res.save_to_img(save_path=img_output_path)
            print(f"可视化图片已保存至: {img_output_path}")

            # 保存JSON结果
            json_output_path = os.path.join(output_dir, f"result_{os.path.splitext(os.path.basename(image_path))[0]}.json")
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(filtered_result, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
            print(f"检测结果已保存至: {json_output_path}")
            
            # 如果启用对话框合并
            if merge_dialogs:
                print("正在合并对话框...")
                merged_result = merge_dialog_boxes(filtered_result)
                
                # 保存合并后的结果
                merged_json_path = os.path.join(output_dir, f"merged_{os.path.splitext(os.path.basename(image_path))[0]}.json")
                with open(merged_json_path, "w", encoding="utf-8") as f:
                    json.dump(merged_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                print(f"合并结果已保存至: {merged_json_path}")
                
                # 显示合并信息
                merge_info = visualize_merged_boxes(merged_result)
                print(f"\n{merge_info}")
                
                # 生成只显示对话框的图像
                try:
                    dialog_img_path = create_dialog_only_image(image_path, merged_json_path)
                    print(f"对话框专用图像已保存至: {dialog_img_path}")
                except ImportError:
                    print("提示: 安装 opencv-python 可生成对话框专用图像")
                except Exception as e:
                    print(f"生成对话框图像时出错: {e}")
                
                filtered_result['merged_result'] = merged_result
        else:
            print(f"未检测到置信度高于 {confidence_threshold} 的文本区域")

    return filtered_results


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数组"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="繁体中文文本检测脚本")
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument("--confidence", "-c", type=float, default=0.75, help="置信度阈值 (默认: 0.75)")
    parser.add_argument("--output", "-o", default="output", help="输出目录 (默认: output)")
    parser.add_argument("--merge", "-m", action="store_true", help="合并可能的对话框区域")

    args = parser.parse_args()

    try:
        # 执行文本检测
        results = detect_text_with_confidence(
            image_path=args.image_path, 
            confidence_threshold=args.confidence, 
            output_dir=args.output,
            merge_dialogs=args.merge
        )

        if results:
            print(f"\n检测完成！共处理 {len(results)} 个图片")
            print(f"结果保存在目录: {args.output}")
        else:
            print("未检测到符合条件的文本区域")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
