"""
智能体主处理器
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..config.settings import AgentConfig
from ..workflows.story_processor import StoryProcessorWorkflow

logger = logging.getLogger(__name__)


class AgentProcessor:
    """智能体主处理器"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        初始化智能体处理器
        
        Args:
            config: 智能体配置
        """
        self.config = config or AgentConfig.default()
        self.story_workflow = StoryProcessorWorkflow(self.config)
        
        logger.info("智能体处理器初始化完成")
    
    def process_story_from_folder(self, input_folder: str) -> Dict[str, Any]:
        """
        从文件夹处理故事
        
        Args:
            input_folder: 输入图像文件夹路径
        
        Returns:
            处理结果
        """
        logger.info(f"开始处理故事，输入文件夹: {input_folder}")
        
        # 验证输入文件夹
        folder_path = Path(input_folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"输入文件夹不存在: {input_folder}")
        
        if not folder_path.is_dir():
            raise ValueError(f"输入路径不是文件夹: {input_folder}")
        
        # 执行故事处理工作流
        result = self.story_workflow.process(input_folder)
        
        if result["status"] == "success":
            logger.info("故事处理完成")
        else:
            logger.error(f"故事处理失败: {result.get('error')}")
        
        return result
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        获取工作流状态信息
        
        Returns:
            状态信息
        """
        return {
            "config": {
                "llm_model": self.config.llm.model_name,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens
            },
            "features": {
                "langgraph_available": hasattr(self.story_workflow, 'workflow') and self.story_workflow.workflow is not None,
                "ocr_integration": True,
                "llm_integration": True
            }
        }