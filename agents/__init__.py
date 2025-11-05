"""
Agents - 智能体模块

基于LangGraph的智能工作流，支持：
- OCR对话框处理集成
- LLM对话脚本生成
- 故事内容总结
- 多步骤工作流管理
"""

__version__ = "1.0.0"
__author__ = "Agents Team"
__description__ = "Intelligent agents with LangGraph workflows for story processing"

from .core.agent_processor import AgentProcessor
from .workflows.story_processor import StoryProcessorWorkflow
from .config.settings import AgentConfig

__all__ = [
    "AgentProcessor",
    "StoryProcessorWorkflow", 
    "AgentConfig"
]