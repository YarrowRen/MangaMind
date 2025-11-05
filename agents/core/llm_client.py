"""
LLM客户端核心组件
"""

import logging
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..config.settings import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM客户端"""
    
    def __init__(self, config: LLMConfig):
        """
        初始化LLM客户端
        
        Args:
            config: LLM配置
        """
        self.config = config
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """初始化OpenAI客户端"""
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装OpenAI库: pip install openai")
        
        if not self.config.api_key:
            raise ValueError("请设置XAI_API_KEY环境变量或在配置中提供API密钥")
        
        client_kwargs = {
            "api_key": self.config.api_key,
            "base_url": self.config.base_url
        }
        
        self.client = OpenAI(**client_kwargs)
        logger.info(f"LLM客户端初始化完成，模型: {self.config.model_name}")
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        生成文本
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
        
        Returns:
            生成的文本
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            messages.append({
                "role": "user", 
                "content": prompt
            })
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            content = response.choices[0].message.content
            
            # 统计token使用情况
            input_chars = len(prompt) + (len(system_prompt) if system_prompt else 0)
            logger.info(f"LLM生成完成，输入字符数: {input_chars}, 输出字符数: {len(content)}")
            
            # 检查是否可能因为长度限制被截断
            if len(content) >= self.config.max_tokens * 3:  # 估算3字符/token
                logger.warning(f"输出可能被截断，当前长度: {len(content)}, 最大tokens: {self.config.max_tokens}")
            
            return content
            
        except Exception as e:
            logger.error(f"LLM生成失败: {e}")
            raise
    
    def generate_dialogue_script(self, json_data: str) -> str:
        """
        生成对话脚本
        
        Args:
            json_data: OCR对话数据的JSON字符串
        
        Returns:
            对话脚本
        """
        # 检查输入长度
        input_length = len(json_data)
        logger.info(f"JSON输入长度: {input_length} 字符")
        
        # 如果JSON过长，考虑警告
        if input_length > 50000:  # 50K字符阈值
            logger.warning(f"JSON输入较长 ({input_length} 字符)，可能影响LLM处理效果")
        
        system_prompt = """你是一个专业的对话脚本分析师，专门处理漫画或图像中的对话内容。"""
        
        user_prompt = f"""你的任务是根据提供的JSON文件，划分不同角色对话以及对话顺序，输出对话脚本。在处理过程中，需要忽略可能存在的背景文字信息（如商店名等无关内容），并且要处理可能不准确的对话顺序。

在处理JSON文件时，请遵循以下规则：
1. 仔细检查每个对话项，判断其是否为有效的对话内容，忽略无关的背景文字信息。（旁白对话不属于无关信息）
2. 根据sequence_id以及对话的逻辑关系，调整可能不准确的对话顺序。
3. 识别不同角色的对话，将属于同一角色的对话进行归类。可以辨别姓名的角色正确处理其姓名，没有足够信息获取姓名的角色用序号方式进行表示区分

首先，在<思考>标签中详细分析JSON文件内容，说明你是如何判断对话内容的有效性、调整对话顺序以及划分角色的。然后，在<对话脚本>标签中输出最终的对话脚本，脚本应清晰呈现不同角色的对话以及对话顺序。

<思考>
[在此详细分析JSON文件内容]
</思考>

<对话脚本>
[在此输出对话脚本]
</对话脚本>

以下是JSON文件内容：
{json_data}"""
        
        return self.generate(user_prompt, system_prompt)
    
    def generate_story_summary(self, dialogue_script: str) -> str:
        """
        生成故事总结
        
        Args:
            dialogue_script: 对话脚本
        
        Returns:
            故事总结
        """
        system_prompt = """你是一个专业的故事总结专家，能够根据对话脚本创作出连贯的故事内容。"""
        
        user_prompt = f"""你的任务是利用给定的角色对话脚本，以连贯且符合故事氛围的自然语言复述整个故事内容。

以下是对话脚本：
<对话脚本>
{dialogue_script}
</对话脚本>

在复述故事时，请遵循以下指南：
1. 保持故事的连贯性和逻辑性。
2. 语言表达要自然流畅，符合故事的氛围。
3. 完整涵盖对话脚本中的所有关键信息。

请在<story>标签内写下你的复述内容。"""
        
        return self.generate(user_prompt, system_prompt)