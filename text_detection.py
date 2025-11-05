#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用文本检测脚本
使用PaddleOCR检测图片中的文本区域，支持多语言文本检测，过滤高置信度结果
"""

import argparse
import json
import os
import sys

import numpy as np
from paddleocr import PaddleOCR, TextDetection

from dialog_merger import merge_dialog_boxes, visualize_merged_boxes, poly_to_rect
from visualize_dialogs import create_dialog_only_image


def create_dialog_sequence(merged_result):
    """
    创建按阅读顺序排列的对话序列
    按从右上角到左下角的顺序排列（以对话框右上顶点坐标为准）
    
    Args:
        merged_result: 合并后的OCR结果
    
    Returns:
        dict: 包含对话序列的结果
    """
    dialog_sequence = []
    
    # 收集所有对话框（合并的和单独的）
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


def find_nearby_ocr_texts(rect, all_rects, used_indices, max_distance=50):
    """查找附近的OCR文本框（更严格的条件）"""
    nearby_indices = []
    
    # 使用更小的扩展比例
    expanded_rect = rect.expand(expand_ratio=1.3)
    
    for other_rect, original_index in all_rects:
        if original_index in used_indices:
            continue
            
        # 使用更严格的条件：优先考虑重叠，距离阈值更小
        if expanded_rect.collision(other_rect):
            nearby_indices.append(original_index)
        elif rect.distance_to(other_rect) <= max_distance:
            # 额外检查：如果距离很近，还要检查是否在合理的相对位置
            # 例如，垂直或水平方向上比较接近
            rect1_center_x = (rect.x0 + rect.x1) / 2
            rect1_center_y = (rect.y0 + rect.y1) / 2
            rect2_center_x = (other_rect.x0 + other_rect.x1) / 2
            rect2_center_y = (other_rect.y0 + other_rect.y1) / 2
            
            # 计算水平和垂直距离
            h_distance = abs(rect1_center_x - rect2_center_x)
            v_distance = abs(rect1_center_y - rect2_center_y)
            
            # 如果一个方向距离很近，另一个方向距离合理，则认为可以合并
            if (h_distance <= max_distance * 1.5 and v_distance <= rect.h * 0.8) or \
               (v_distance <= max_distance * 1.5 and h_distance <= rect.w * 0.8):
                nearby_indices.append(original_index)
    
    return nearby_indices


def merge_ocr_dialog_boxes(ocr_result, max_distance=50, min_group_size=2):
    """
    合并OCR对话框文本区域，支持文字排序
    
    Args:
        ocr_result: OCR结果，包含dt_polys, dt_scores, dt_texts
        max_distance: 最大合并距离
        min_group_size: 最小群组大小
    
    Returns:
        合并后的结果
    """
    dt_polys = ocr_result['dt_polys']
    dt_scores = ocr_result['dt_scores']
    dt_texts = ocr_result['dt_texts']
    
    if not dt_polys:
        return ocr_result
    
    # 转换为矩形对象，并保存文字信息
    rectangles = []
    for i, poly in enumerate(dt_polys):
        rect = poly_to_rect(poly)
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
        def find_connected_texts(current_rect, current_indices):
            nearby_indices = find_nearby_ocr_texts(current_rect, rectangles, used_indices, max_distance)
            
            for nearby_idx in nearby_indices:
                if nearby_idx not in used_indices:
                    current_indices.append(nearby_idx)
                    used_indices.add(nearby_idx)
                    
                    # 递归查找与新加入文本框相邻的文本框
                    nearby_rect = next(r for r, idx in rectangles if idx == nearby_idx)
                    find_connected_texts(nearby_rect, current_indices)
        
        find_connected_texts(rect, current_group)
        
        # 只保留群组大小满足要求的或单独的文本框
        if len(current_group) >= min_group_size or min_group_size == 1:
            merged_groups.append(current_group)
        else:
            # 对于小群组，将每个文本框作为单独的群组
            for idx in current_group:
                merged_groups.append([idx])
    
    # 构建合并后的结果
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
        "dialog_boxes": []  # 添加这个字段以兼容可视化函数
    }
    
    for group_indices in merged_groups:
        if len(group_indices) >= min_group_size:
            # 对话框合并：按从右到左排序文字
            group_data = []
            for idx in group_indices:
                poly = dt_polys[idx]
                rect = poly_to_rect(poly)
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
            
            # 同时添加到dialog_boxes以兼容可视化函数
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
                
                # 同时添加到dialog_boxes以兼容可视化函数
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


def detect_text_with_confidence(
    image_path, confidence_threshold=0.75, output_dir="output", merge_dialogs=False, enable_ocr=False, ocr_lang="chinese_cht"
):
    """
    检测图片中的文本区域，过滤高置信度结果

    Args:
        image_path (str): 输入图片路径
        confidence_threshold (float): 置信度阈值，默认0.75
        output_dir (str): 输出目录
        merge_dialogs (bool): 是否合并对话框，默认False
        enable_ocr (bool): 是否启用OCR文字识别，默认False
        ocr_lang (str): OCR识别语言，默认chinese_cht（繁体中文）

    Returns:
        dict: 检测结果
    """
    # 检查输入文件是否存在
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图片不存在: {image_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 根据是否启用OCR选择不同的模型
    if enable_ocr:
        print("正在初始化OCR模型...")
        # 尝试禁用可能导致旋转的参数
        model = PaddleOCR(
            lang=ocr_lang,
            text_det_limit_type="max",
            text_det_limit_side_len=960,
            use_doc_orientation_classify=False,  # 禁用文档方向分类
            use_doc_unwarping=False,  # 禁用文档矫正
        )
    else:
        print("正在初始化文本检测模型...")
        model = TextDetection(
            model_name="PP-OCRv5_server_det",
            device="cpu",  # 可根据需要改为"gpu"
            box_thresh=confidence_threshold,  # 设置检测框置信度阈值
        )

    # 进行文本检测或OCR
    print(f"正在{'识别' if enable_ocr else '检测'}图片: {image_path}")

    if enable_ocr:
        # OCR模式：返回格式为 [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence)]
        ocr_results = model.predict(image_path)

        # 处理OCR结果 - PaddleOCR返回格式不同
        filtered_results = []

        if ocr_results and len(ocr_results) > 0:
            # 检查是否是predict方法返回的结果对象
            if hasattr(ocr_results[0], "json"):
                # predict方法返回的结果
                for res in ocr_results:
                    result_data = res.json

                    if "res" in result_data:
                        input_path = result_data["res"]["input_path"]
                        res_data = result_data["res"]

                        # 检查OCR结果结构 - 新的格式包含rec_texts, rec_scores, rec_polys
                        if "rec_texts" in res_data and "rec_scores" in res_data and "rec_polys" in res_data:
                            rec_texts = res_data["rec_texts"]
                            rec_scores = res_data["rec_scores"]
                            rec_polys = res_data["rec_polys"]

                            dt_polys = []
                            dt_scores = []
                            dt_texts = []

                            # 过滤高置信度结果
                            for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                                if score > confidence_threshold:
                                    dt_polys.append(poly)
                                    dt_scores.append(score)
                                    dt_texts.append(text)

                            if dt_polys:
                                filtered_result = {
                                    "input_path": input_path,
                                    "confidence_threshold": confidence_threshold,
                                    "total_detections": len(rec_texts),
                                    "high_confidence_detections": len(dt_polys),
                                    "dt_polys": dt_polys,
                                    "dt_scores": dt_scores,
                                    "dt_texts": dt_texts,
                                    "ocr_enabled": True,
                                }

                                filtered_results.append(filtered_result)

                                print(
                                    f"OCR识别到 {len(rec_texts)} 个文本区域，其中 {len(dt_polys)} 个置信度高于 {confidence_threshold}"
                                )

                                # 保存可视化图片
                                img_output_path = os.path.join(output_dir, f"ocr_{os.path.basename(image_path)}")
                                res.save_to_img(save_path=img_output_path)
                                print(f"OCR可视化图片已保存至: {img_output_path}")

                                # 保存OCR JSON结果
                                json_output_path = os.path.join(
                                    output_dir, f"ocr_{os.path.splitext(os.path.basename(image_path))[0]}.json"
                                )
                                with open(json_output_path, "w", encoding="utf-8") as f:
                                    json.dump(filtered_result, f, ensure_ascii=False, indent=4, cls=NumpyEncoder)
                                print(f"OCR结果已保存至: {json_output_path}")
                                
                                # 如果启用对话框合并
                                if merge_dialogs:
                                    print("正在合并OCR对话框...")
                                    # 使用更严格的合并参数
                                    merged_result = merge_ocr_dialog_boxes(filtered_result, max_distance=40, min_group_size=2)

                                    # 保存合并后的结果
                                    merged_json_path = os.path.join(
                                        output_dir, f"merged_ocr_{os.path.splitext(os.path.basename(image_path))[0]}.json"
                                    )
                                    with open(merged_json_path, "w", encoding="utf-8") as f:
                                        json.dump(merged_result, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                                    print(f"OCR合并结果已保存至: {merged_json_path}")

                                    # 生成对话序列
                                    dialog_sequence = create_dialog_sequence(merged_result)
                                    
                                    # 保存对话序列
                                    sequence_json_path = os.path.join(
                                        output_dir, f"dialog_sequence_{os.path.splitext(os.path.basename(image_path))[0]}.json"
                                    )
                                    with open(sequence_json_path, "w", encoding="utf-8") as f:
                                        json.dump(dialog_sequence, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                                    print(f"对话序列已保存至: {sequence_json_path}")

                                    # 显示合并信息
                                    merge_info = visualize_merged_boxes(merged_result)
                                    print(f"\\n{merge_info}")
                                    
                                    # 显示对话序列信息
                                    print(f"\\n对话序列信息:")
                                    print(f"总对话数: {dialog_sequence['total_dialogs']}")
                                    print(f"排序方式: {dialog_sequence['reading_order']}")
                                    for dialog in dialog_sequence['dialog_sequence'][:3]:  # 显示前3个
                                        print(f"  {dialog['sequence_id']}. [{dialog['dialog_type']}] {dialog['dialog_text'][:20]}{'...' if len(dialog['dialog_text']) > 20 else ''}")
                                    if len(dialog_sequence['dialog_sequence']) > 3:
                                        print(f"  ... 还有 {len(dialog_sequence['dialog_sequence']) - 3} 个对话")

                                    # 生成只显示对话框的图像
                                    try:
                                        dialog_img_path = create_dialog_only_image(image_path, merged_json_path)
                                        print(f"OCR对话框专用图像已保存至: {dialog_img_path}")
                                    except ImportError:
                                        print("提示: 安装 opencv-python 可生成对话框专用图像")
                                    except Exception as e:
                                        print(f"生成OCR对话框图像时出错: {e}")

                                    filtered_result["merged_result"] = merged_result
                            else:
                                print(f"未检测到置信度高于 {confidence_threshold} 的文本区域")
                        else:
                            print("OCR结果格式不符合预期")
                    else:
                        print("OCR结果格式错误")
            else:
                print("OCR结果格式不支持")
    else:
        # 检测模式
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
                    "ocr_enabled": False,
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

                filtered_result["merged_result"] = merged_result
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
    parser = argparse.ArgumentParser(description="通用文本检测脚本")
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument("--confidence", "-c", type=float, default=0.75, help="置信度阈值 (默认: 0.75)")
    parser.add_argument("--output", "-o", default="output", help="输出目录 (默认: output)")
    parser.add_argument("--merge", "-m", action="store_true", help="合并可能的对话框区域")
    parser.add_argument("--ocr", action="store_true", help="启用OCR文字识别（默认仅检测）")
    parser.add_argument("--lang", "-l", default="chinese_cht", help="OCR识别语言 (默认: chinese_cht)")

    args = parser.parse_args()

    try:
        # 执行文本检测
        results = detect_text_with_confidence(
            image_path=args.image_path,
            confidence_threshold=args.confidence,
            output_dir=args.output,
            merge_dialogs=args.merge,
            enable_ocr=args.ocr,
            ocr_lang=args.lang,
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
