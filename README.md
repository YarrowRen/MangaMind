# 繁体中文文本检测脚本使用说明

## 功能描述
本脚本基于PaddleOCR实现，用于检测图片中的文本区域，过滤置信度高于0.9的结果，并保存可视化图片和JSON格式的检测结果。

## 安装依赖

首先安装PaddleOCR：

```bash
pip install paddleocr
```

## 使用方法

### 基本用法
```bash
python detect_traditional_chinese.py <图片路径>
```

### 完整参数
```bash
python detect_traditional_chinese.py <图片路径> [选项]

选项:
  -c, --confidence  置信度阈值 (默认: 0.9)
  -o, --output      输出目录 (默认: output)
  -h, --help        显示帮助信息
```

### 使用示例

1. **基本检测（使用默认参数）**
   ```bash
   python detect_traditional_chinese.py image.jpg
   ```

2. **自定义置信度阈值**
   ```bash
   python detect_traditional_chinese.py image.jpg --confidence 0.8
   ```

3. **指定输出目录**
   ```bash
   python detect_traditional_chinese.py image.jpg --output results
   ```

4. **完整参数示例**
   ```bash
   python detect_traditional_chinese.py image.jpg -c 0.95 -o my_results
   ```

## 输出结果

脚本会在指定的输出目录中生成以下文件：

1. **可视化图片**: `detected_<原图片名>`
   - 在原图上标注检测到的文本区域边框

2. **JSON结果文件**: `result_<图片名>.json`
   - 包含检测结果的详细信息

### JSON结果格式
```json
{
    "input_path": "输入图片路径",
    "confidence_threshold": 0.9,
    "total_detections": 10,
    "high_confidence_detections": 5,
    "dt_polys": [
        [
            [x1, y1], [x2, y2], [x3, y3], [x4, y4]
        ]
    ],
    "dt_scores": [0.95, 0.92, 0.98]
}
```

字段说明：
- `input_path`: 输入图片路径
- `confidence_threshold`: 使用的置信度阈值
- `total_detections`: 总检测区域数量
- `high_confidence_detections`: 高置信度区域数量
- `dt_polys`: 文本区域的四边形坐标点
- `dt_scores`: 对应区域的置信度分数

## 模型说明

脚本使用 `PP-OCRv5_server_det` 高精度文本检测模型：
- 检测精度: 83.8% (Hmean)
- 适合服务端部署，精度较高
- 支持多种文字（包括繁体中文）

## 注意事项

1. 首次运行时，PaddleOCR会自动下载模型文件，可能需要一些时间
2. 如果需要GPU加速，请安装GPU版本的PaddlePaddle
3. 脚本默认使用CPU推理，如需GPU推理，请修改脚本中的`device="cpu"`为`device="gpu"`
4. 确保输入图片格式支持（JPG、PNG等常见格式）

## 常见问题

**Q: 如何提高检测精度？**
A: 可以降低置信度阈值（如0.8），或使用更高分辨率的图片

**Q: 如何加速检测？**
A: 可以将模型改为`PP-OCRv5_mobile_det`移动端模型，或使用GPU推理

**Q: 支持批量处理吗？**
A: 当前版本支持单张图片处理，如需批量处理可以结合shell脚本使用