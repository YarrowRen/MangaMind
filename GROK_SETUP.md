# Grok API 配置指南

## 环境配置

### 1. 设置API密钥

```bash
# 设置Grok API密钥
export XAI_API_KEY="your-xai-api-key"

# 验证设置
echo $XAI_API_KEY
```

### 2. 安装依赖

```bash
# 安装智能体模块依赖
pip install openai>=1.0.0 langgraph>=0.0.40

# 安装OCR模块依赖（如果需要）
pip install paddleocr paddlepaddle opencv-python
```

## 使用方法

### 基本用法

```bash
# 设置API密钥
export XAI_API_KEY="your-xai-api-key"

# 运行故事生成
python story_agent.py input_folder
```

### 高级配置

```bash
# 使用不同的Grok模型
python story_agent.py input_folder --model grok-2

# 调整温度参数（创造性）
python story_agent.py input_folder --temperature 0.3

# 增加最大输出长度
python story_agent.py input_folder --max-tokens 4000

# 查看状态信息
python story_agent.py input_folder --status
```

## 模型选择

支持的Grok模型：
- `grok-4-fast-non-reasoning` (默认) - 快速推理模型
- `grok-2` - 标准Grok-2模型
- `grok-beta` - 测试版本

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `grok-4-fast-non-reasoning` | Grok模型名称 |
| `base_url` | `https://api.x.ai/v1` | Grok API端点 |
| `temperature` | `0.7` | 创造性参数 (0-1) |
| `max_tokens` | `2000` | 最大输出长度 |
| `timeout` | `30` | 请求超时时间(秒) |

## 环境变量

- `XAI_API_KEY`: Grok API密钥 (必需)

## 输出文件

处理完成后会生成：
- `dialogue_script.md` - 角色对话脚本
- `story_summary.md` - 故事内容总结
- `complete_report.md` - 完整处理报告

## 错误排查

1. **API密钥错误**: 确保设置了正确的 `XAI_API_KEY`
2. **网络连接**: 确保能访问 `https://api.x.ai/v1`
3. **模型不存在**: 检查模型名称是否正确
4. **配额不足**: 检查Grok API配额和余额