"""
日志工具模块
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from ..config.settings import LoggingConfig


class Logger:
    """日志管理器"""
    
    _instances = {}
    
    def __new__(cls, name: str, config: Optional[LoggingConfig] = None):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(self, name: str, config: Optional[LoggingConfig] = None):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self.name = name
        self.config = config or LoggingConfig()
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志记录器"""
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置格式器
        formatter = logging.Formatter(self.config.format)
        
        # 控制台处理器
        if self.config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if self.config.file_path:
            log_dir = Path(self.config.file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                self.config.file_path,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, *args, **kwargs):
        """调试级别日志"""
        self.logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """信息级别日志"""
        self.logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """警告级别日志"""
        self.logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """错误级别日志"""
        self.logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """严重错误级别日志"""
        self.logger.critical(message, *args, **kwargs)


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> Logger:
    """获取日志记录器"""
    return Logger(name, config)