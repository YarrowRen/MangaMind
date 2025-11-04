# 对话框文本合并系统使用说明

## 功能概述

本系统基于PaddleOCR和矩形碰撞检测算法，实现了以下功能：

1. **文本检测**: 检测图片中的文本区域，过滤高置信度结果
2. **对话框合并**: 将空间上相邻的文本框合并为对话框区域
3. **可视化显示**: 生成包含检测框和合并结果的可视化图像

## 文件说明

- `detect_traditional_chinese.py` - 主检测脚本（集成合并功能）
- `dialog_merger.py` - 对话框合并算法实现
- `visualize_dialogs.py` - 可视化脚本
- `README_dialog_merger.md` - 本说明文件

## 使用方法

### 1. 基本文本检测

```bash
# 基本检测（不合并）
python detect_traditional_chinese.py input/image.jpg

# 自定义置信度
python detect_traditional_chinese.py input/image.jpg --confidence 0.8
```

### 2. 带对话框合并的检测（推荐）

```bash
# 启用对话框合并（自动生成所有可视化文件）
python detect_traditional_chinese.py input/image.jpg --merge

# 完整参数示例
python detect_traditional_chinese.py input/image.jpg --merge --confidence 0.75 --output results
```

### 3. 单独使用合并功能

```bash
# 对现有检测结果进行合并
python dialog_merger.py output/result_test.json --verbose

# 自定义合并参数
python dialog_merger.py result.json --distance 200 --min-size 3
```

### 4. 可视化结果

```bash
# 生成只显示对话框的图像（推荐）
python visualize_dialogs.py input/image.jpg output/merged_result.json --dialogs-only

# 生成对比图像（原图|检测结果|合并结果）
python visualize_dialogs.py input/image.jpg output/merged_result.json --comparison

# 生成完整可视化（包含原始文本框和对话框）
python visualize_dialogs.py input/image.jpg output/merged_result.json
```

## 输出文件说明

使用 `--merge` 参数时，系统会自动生成以下文件：

- `detected_*.jpg` - 原始文本检测可视化
- `result_*.json` - 文本检测结果数据
- `merged_*.json` - 对话框合并结果数据
- `dialogs_only_*.jpg` - **只显示对话框边界的图像（主要输出）**
- `comparison_*.jpg` - 三图对比（需要单独生成）

## 算法原理

### 矩形碰撞检测

使用改进的矩形碰撞算法：

1. **扩展检测区域**: 将每个文本框按比例扩展，增加合并的容错性
2. **距离计算**: 基于中心点距离判断文本框的空间关系
3. **递归合并**: 从大文本框开始，递归查找相邻的文本框

### 合并策略

- **距离阈值**: 默认150像素，可通过`--distance`参数调整
- **最小群组**: 默认需要2个以上文本框才形成对话框，可通过`--min-size`调整
- **优先级**: 按文本框面积排序，优先处理大文本框

## 输出格式

### 检测结果 (result_*.json)
```json
{
    "input_path": "图片路径",
    "confidence_threshold": 0.75,
    "total_detections": 11,
    "high_confidence_detections": 11,
    "dt_polys": [...],  // 文本框坐标
    "dt_scores": [...]  // 置信度分数
}
```

### 合并结果 (merged_*.json)
```json
{
    "input_path": "图片路径",
    "original_dt_polys": [...],     // 原始文本框
    "original_dt_scores": [...],    // 原始置信度
    "merged_groups": [[0,1,2], [3,4]], // 合并的文本框索引组
    "dialog_boxes": [
        {
            "indices": [0, 1, 2],           // 包含的文本框索引
            "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], // 对话框边界
            "individual_polys": [...],       // 单个文本框坐标
            "individual_scores": [...],      // 单个文本框置信度
            "avg_score": 0.85,              // 平均置信度
            "text_count": 3                 // 文本框数量
        }
    ]
}
```

## 参数说明

### detect_traditional_chinese.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image_path` | 输入图片路径 | 必需 |
| `--confidence, -c` | 置信度阈值 | 0.75 |
| `--output, -o` | 输出目录 | output |
| `--merge, -m` | 启用对话框合并 | False |

### dialog_merger.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `json_file` | 检测结果JSON文件 | 必需 |
| `--output, -o` | 输出文件路径 | 自动生成 |
| `--distance, -d` | 最大合并距离 | 150 |
| `--min-size, -m` | 最小群组大小 | 2 |
| `--verbose, -v` | 显示详细信息 | False |

### visualize_dialogs.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image_path` | 原始图片路径 | 必需 |
| `merged_json` | 合并结果JSON | 必需 |
| `--output, -o` | 输出图片路径 | 自动生成 |
| `--comparison, -c` | 创建对比图像 | False |
| `--dialogs-only, -d` | **只显示对话框边界** | False |

## 示例工作流

```bash
# 1. 完整检测和合并（推荐方式）
python detect_traditional_chinese.py input/manga_page.jpg --merge --confidence 0.8

# 输出文件:
# output/detected_manga_page.jpg    - 原始检测可视化
# output/result_manga_page.json     - 检测结果
# output/merged_manga_page.json     - 合并结果
# output/dialogs_only_manga_page.jpg - 对话框专用图像 ⭐

# 2. 可选：生成额外的对比图
python visualize_dialogs.py input/manga_page.jpg output/merged_manga_page.json --comparison
```

## 关键特性

### ⭐ 对话框专用图像
- 文件名：`dialogs_only_*.jpg`
- 特点：在原图上只显示合并后的对话框边界
- 用途：清晰展示检测到的对话区域，适合后续处理

### 🔍 智能合并算法
- 自动识别空间相邻的文本框
- 支持不规则对话框形状
- 可调节合并敏感度

### 📊 多种可视化选项
- 对话框专用图像（主要输出）
- 三图对比（原图|检测|合并）
- 完整可视化（所有检测框+对话框）

## 应用场景

1. **漫画文本检测**: 识别漫画中的对话框和文字气泡
2. **文档分析**: 检测文档中的文本块并按区域分组
3. **UI界面分析**: 识别界面中的文本元素并按功能分组

## 性能调优

- **提高召回率**: 降低置信度阈值 (`--confidence 0.6`)
- **减少误合并**: 减小合并距离 (`--distance 100`)
- **处理密集文本**: 增加最小群组大小 (`--min-size 3`)
- **加速处理**: 使用GPU (`device="gpu"` 在代码中修改)

## 依赖安装

```bash
# 基本依赖
pip install paddleocr

# 可视化依赖
pip install opencv-python

# 完整安装
pip install paddleocr opencv-python numpy
```

## 注意事项

1. 首次运行会下载PaddleOCR模型文件
2. 需要安装OpenCV用于可视化功能
3. 合并算法基于空间距离，可能需要根据具体图片调整参数
4. 建议在高分辨率图片上使用以获得更好的检测效果
5. **主要输出是 `dialogs_only_*.jpg` 文件，显示合并后的对话框边界**