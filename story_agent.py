#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Story Agent - 故事处理智能体主入口
基于LangGraph的端到端故事处理工作流
"""

import argparse
import sys
import logging
from pathlib import Path

from agents.core.agent_processor import AgentProcessor
from agents.config.settings import AgentConfig, LLMConfig


def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_config_from_args(args) -> AgentConfig:
    """从命令行参数创建配置"""
    llm_config = LLMConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    config = AgentConfig.default()
    config.llm = llm_config
    
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Story Agent - 基于LangGraph的故事处理智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s input_folder                     # 处理文件夹中的图像生成故事
  %(prog)s input_folder --model grok-2     # 使用Grok-2模型
  %(prog)s input_folder --temperature 0.3  # 调整创造性
        """
    )
    
    # 基本参数
    parser.add_argument("input_folder", help="输入图像文件夹路径")
    
    # LLM参数
    parser.add_argument("--model", "-m", default="grok-4-fast-non-reasoning", 
                       help="LLM模型名称 (默认: grok-4-fast-non-reasoning)")
    parser.add_argument("--temperature", "-t", type=float, default=0.7,
                       help="LLM温度参数 (默认: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=2000,
                       help="最大生成token数 (默认: 2000)")
    
    # 系统参数
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别")
    parser.add_argument("--status", action="store_true", 
                       help="显示智能体状态信息")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # 创建配置
        config = create_config_from_args(args)
        
        # 初始化智能体处理器
        agent = AgentProcessor(config)
        
        # 显示状态信息
        if args.status:
            status = agent.get_workflow_status()
            logger.info("智能体状态信息:")
            logger.info(f"  LLM模型: {status['config']['llm_model']}")
            logger.info(f"  温度参数: {status['config']['temperature']}")
            logger.info(f"  最大tokens: {status['config']['max_tokens']}")
            logger.info(f"  LangGraph可用: {status['features']['langgraph_available']}")
            logger.info(f"  OCR集成: {status['features']['ocr_integration']}")
            logger.info(f"  LLM集成: {status['features']['llm_integration']}")
        
        # 验证输入文件夹
        if not Path(args.input_folder).exists():
            logger.error(f"输入文件夹不存在: {args.input_folder}")
            sys.exit(1)
        
        # 处理故事
        logger.info("开始故事处理工作流...")
        result = agent.process_story_from_folder(args.input_folder)
        
        if result["status"] == "success":
            logger.info("🎉 故事处理完成！")
            
            # 显示处理结果摘要
            step_outputs = result.get("step_outputs", {})
            for step, output in step_outputs.items():
                if output.get("status") == "success":
                    logger.info(f"✅ {step}: 成功")
                    if step == "ocr":
                        logger.info(f"   处理页面: {output.get('total_pages', 0)}")
                        logger.info(f"   识别对话: {output.get('total_dialogs', 0)}")
                    elif step == "output":
                        logger.info(f"   输出目录: {output.get('output_dir')}")
                        logger.info(f"   生成文件: {', '.join(output.get('files', []))}")
                else:
                    logger.error(f"❌ {step}: 失败 - {output.get('error', '未知错误')}")
            
            if result.get("has_dialogue_script"):
                logger.info("📝 对话脚本已生成")
            
            if result.get("has_story_summary"):
                logger.info("📖 故事总结已生成")
                
        else:
            logger.error(f"❌ 故事处理失败: {result.get('error', '未知错误')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理失败: {e}")
        if args.log_level == "DEBUG":
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()