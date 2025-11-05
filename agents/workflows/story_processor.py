"""
故事处理工作流 - 基于LangGraph
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict
from dataclasses import asdict

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.state import CompiledStateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # 创建简单的替代类型
    class StateGraph:
        def __init__(self, schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def compile(self): return None
    START = "START"
    END = "END"

from ..core.llm_client import LLMClient
from ..config.settings import AgentConfig

# 使用相对路径导入OCR模块
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ocr_dialog.core.processor import OCRDialogProcessor
from ocr_dialog.config.settings import AppConfig

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """工作流状态定义"""
    input_folder: str
    ocr_results: Optional[List[Dict[str, Any]]]
    batch_dialogue_data: Optional[str]
    dialogue_script: Optional[str]
    story_summary: Optional[str]
    error: Optional[str]
    step_outputs: Dict[str, Any]


class StoryProcessorWorkflow:
    """故事处理工作流"""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        初始化工作流
        
        Args:
            config: 智能体配置
        """
        self.config = config or AgentConfig.default()
        self.llm_client = LLMClient(self.config.llm)
        self.ocr_processor = None
        self.workflow = None
        
        if LANGGRAPH_AVAILABLE:
            self._build_workflow()
        else:
            logger.warning("LangGraph不可用，将使用简化模式")
        
        logger.info("故事处理工作流初始化完成")
    
    def _build_workflow(self):
        """构建LangGraph工作流"""
        # 创建状态图
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("ocr_processing", self._ocr_processing_node)
        workflow.add_node("dialogue_generation", self._dialogue_generation_node)  
        workflow.add_node("story_summarization", self._story_summarization_node)
        workflow.add_node("output_formatting", self._output_formatting_node)
        
        # 添加边
        workflow.add_edge(START, "ocr_processing")
        workflow.add_edge("ocr_processing", "dialogue_generation")
        workflow.add_edge("dialogue_generation", "story_summarization")
        workflow.add_edge("story_summarization", "output_formatting")
        workflow.add_edge("output_formatting", END)
        
        # 编译工作流
        self.workflow = workflow.compile()
        
        logger.info("LangGraph工作流构建完成")
    
    def _get_ocr_processor(self) -> OCRDialogProcessor:
        """获取OCR处理器（懒加载）"""
        if self.ocr_processor is None:
            # 创建OCR配置，启用OCR和对话框合并
            from ocr_dialog.config.settings import OCRConfig, DialogMergerConfig, ProcessingConfig, LoggingConfig
            
            ocr_config = AppConfig(
                ocr=OCRConfig(confidence_threshold=0.75),
                dialog_merger=DialogMergerConfig(),
                processing=ProcessingConfig(
                    merge_dialogs=True,
                    enable_ocr=True,
                    output_formats=["json", "sequence"]
                ),
                logging=LoggingConfig(level="INFO", console_output=False)
            )
            
            self.ocr_processor = OCRDialogProcessor(ocr_config)
        
        return self.ocr_processor
    
    def _ocr_processing_node(self, state: WorkflowState) -> WorkflowState:
        """OCR处理节点"""
        try:
            logger.info(f"开始OCR处理: {state['input_folder']}")
            
            # 使用批处理功能处理整个文件夹
            from batch_process import BatchProcessor
            from ocr_dialog.config.settings import OCRConfig, DialogMergerConfig, ProcessingConfig, LoggingConfig
            
            # 创建批处理配置
            batch_config = AppConfig(
                ocr=OCRConfig(confidence_threshold=0.75),
                dialog_merger=DialogMergerConfig(),
                processing=ProcessingConfig(
                    merge_dialogs=True,
                    enable_ocr=True,
                    output_formats=["json", "sequence"]
                ),
                logging=LoggingConfig(level="INFO", console_output=False)
            )
            
            # 创建临时输出目录
            output_dir = Path(state['input_folder']).parent / "ocr_output"
            
            # 执行批处理
            batch_processor = BatchProcessor(batch_config)
            batch_processor.batch_process_images(state['input_folder'], str(output_dir))
            
            # 读取批处理结果
            batch_result_file = output_dir / "batch_dialogs.json"
            if batch_result_file.exists():
                with open(batch_result_file, 'r', encoding='utf-8') as f:
                    batch_data = json.load(f)
                
                state['batch_dialogue_data'] = json.dumps(batch_data, ensure_ascii=False, indent=2)
                state['step_outputs']['ocr'] = {
                    "status": "success",
                    "total_pages": batch_data.get("total_pages", 0),
                    "total_dialogs": batch_data.get("total_dialogs", 0),
                    "output_dir": str(output_dir)
                }
                
                logger.info(f"OCR处理完成，共处理 {batch_data.get('total_pages', 0)} 页，"
                           f"识别 {batch_data.get('total_dialogs', 0)} 个对话")
            else:
                raise FileNotFoundError("批处理结果文件未找到")
                
        except Exception as e:
            logger.error(f"OCR处理失败: {e}")
            state['error'] = f"OCR处理失败: {str(e)}"
            state['step_outputs']['ocr'] = {"status": "error", "error": str(e)}
        
        return state
    
    def _dialogue_generation_node(self, state: WorkflowState) -> WorkflowState:
        """对话脚本生成节点"""
        try:
            if state.get('error') or not state.get('batch_dialogue_data'):
                return state
            
            logger.info("开始生成对话脚本")
            
            # 调用LLM生成对话脚本
            dialogue_script = self.llm_client.generate_dialogue_script(state['batch_dialogue_data'])
            
            state['dialogue_script'] = dialogue_script
            state['step_outputs']['dialogue'] = {
                "status": "success",
                "script_length": len(dialogue_script)
            }
            
            logger.info(f"对话脚本生成完成，长度: {len(dialogue_script)} 字符")
            
        except Exception as e:
            logger.error(f"对话脚本生成失败: {e}")
            state['error'] = f"对话脚本生成失败: {str(e)}"
            state['step_outputs']['dialogue'] = {"status": "error", "error": str(e)}
        
        return state
    
    def _story_summarization_node(self, state: WorkflowState) -> WorkflowState:
        """故事总结节点"""
        try:
            if state.get('error') or not state.get('dialogue_script'):
                return state
            
            logger.info("开始生成故事总结")
            
            # 调用LLM生成故事总结
            story_summary = self.llm_client.generate_story_summary(state['dialogue_script'])
            
            state['story_summary'] = story_summary
            state['step_outputs']['story'] = {
                "status": "success",
                "summary_length": len(story_summary)
            }
            
            logger.info(f"故事总结生成完成，长度: {len(story_summary)} 字符")
            
        except Exception as e:
            logger.error(f"故事总结生成失败: {e}")
            state['error'] = f"故事总结生成失败: {str(e)}"
            state['step_outputs']['story'] = {"status": "error", "error": str(e)}
        
        return state
    
    def _output_formatting_node(self, state: WorkflowState) -> WorkflowState:
        """输出格式化节点"""
        try:
            if state.get('error'):
                return state
            
            logger.info("开始格式化输出")
            
            # 保存结果到文件
            output_dir = Path(state['input_folder']).parent / "story_output"
            output_dir.mkdir(exist_ok=True)
            
            # 保存对话脚本
            if state.get('dialogue_script'):
                dialogue_file = output_dir / "dialogue_script.md"
                with open(dialogue_file, 'w', encoding='utf-8') as f:
                    f.write("# 对话脚本\n\n")
                    f.write(state['dialogue_script'])
                
                logger.info(f"对话脚本已保存至: {dialogue_file}")
            
            # 保存故事总结
            if state.get('story_summary'):
                story_file = output_dir / "story_summary.md"
                with open(story_file, 'w', encoding='utf-8') as f:
                    f.write("# 故事总结\n\n")
                    f.write(state['story_summary'])
                
                logger.info(f"故事总结已保存至: {story_file}")
            
            # 保存完整报告
            report_file = output_dir / "complete_report.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# 故事处理完整报告\n\n")
                
                # 处理摘要
                f.write("## 处理摘要\n\n")
                for step, result in state['step_outputs'].items():
                    f.write(f"- **{step}**: {result.get('status', 'unknown')}\n")
                f.write("\n")
                
                # 对话脚本
                if state.get('dialogue_script'):
                    f.write("## 对话脚本\n\n")
                    f.write(state['dialogue_script'])
                    f.write("\n\n")
                
                # 故事总结
                if state.get('story_summary'):
                    f.write("## 故事总结\n\n")
                    f.write(state['story_summary'])
                    f.write("\n\n")
            
            state['step_outputs']['output'] = {
                "status": "success",
                "output_dir": str(output_dir),
                "files": ["dialogue_script.md", "story_summary.md", "complete_report.md"]
            }
            
            logger.info(f"输出格式化完成，结果保存在: {output_dir}")
            
        except Exception as e:
            logger.error(f"输出格式化失败: {e}")
            state['error'] = f"输出格式化失败: {str(e)}"
            state['step_outputs']['output'] = {"status": "error", "error": str(e)}
        
        return state
    
    def process(self, input_folder: str) -> Dict[str, Any]:
        """
        处理故事生成工作流
        
        Args:
            input_folder: 输入图像文件夹路径
        
        Returns:
            处理结果
        """
        # 初始化状态
        initial_state: WorkflowState = {
            "input_folder": input_folder,
            "ocr_results": None,
            "batch_dialogue_data": None,
            "dialogue_script": None,
            "story_summary": None,
            "error": None,
            "step_outputs": {}
        }
        
        if LANGGRAPH_AVAILABLE and self.workflow:
            # 使用LangGraph执行工作流
            logger.info("使用LangGraph执行工作流")
            final_state = self.workflow.invoke(initial_state)
        else:
            # 简化模式：顺序执行各个步骤
            logger.info("使用简化模式执行工作流")
            final_state = initial_state
            final_state = self._ocr_processing_node(final_state)
            final_state = self._dialogue_generation_node(final_state)
            final_state = self._story_summarization_node(final_state)
            final_state = self._output_formatting_node(final_state)
        
        # 返回结果摘要
        result = {
            "status": "error" if final_state.get('error') else "success",
            "error": final_state.get('error'),
            "step_outputs": final_state.get('step_outputs', {}),
            "has_dialogue_script": bool(final_state.get('dialogue_script')),
            "has_story_summary": bool(final_state.get('story_summary'))
        }
        
        return result