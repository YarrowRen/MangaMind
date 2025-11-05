"""
智能体配置管理
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class LLMConfig:
    """LLM配置"""
    model_name: str = "grok-4-fast-non-reasoning"
    api_key: Optional[str] = None
    base_url: str = "https://api.x.ai/v1"
    temperature: float = 0.7
    max_tokens: int = 8000
    timeout: int = 60
    
    def __post_init__(self):
        # 从环境变量获取API密钥
        if self.api_key is None:
            self.api_key = os.getenv("XAI_API_KEY")


@dataclass
class DialogueConfig:
    """对话处理配置"""
    ignore_background_text: bool = True
    min_dialogue_length: int = 2
    max_role_count: int = 10
    use_role_names: bool = True


@dataclass
class StoryConfig:
    """故事处理配置"""
    max_story_length: int = 1000
    include_atmosphere: bool = True
    preserve_dialogue_order: bool = True


@dataclass
class WorkflowConfig:
    """工作流配置"""
    enable_logging: bool = True
    log_level: str = "INFO"
    output_format: str = "markdown"
    save_intermediate_results: bool = True


@dataclass
class AgentConfig:
    """智能体总配置"""
    llm: LLMConfig
    dialogue: DialogueConfig
    story: StoryConfig
    workflow: WorkflowConfig
    
    @classmethod
    def default(cls) -> "AgentConfig":
        """创建默认配置"""
        return cls(
            llm=LLMConfig(),
            dialogue=DialogueConfig(),
            story=StoryConfig(),
            workflow=WorkflowConfig()
        )