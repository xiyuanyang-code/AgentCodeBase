"""
AgentCodeBase - A comprehensive toolkit for AI agent development and testing.

This package provides modules for:
- API testing and monitoring (codebase.test)
- Language model interactions (codebase.llm)
- Data generation pipelines (codebase.llm_pipeline)
- Visualization tools (codebase.draw)
- Shared utilities (codebase._utils)

Example usage:
    >>> from codebase import APITester, get_config, get_logger
    >>>
    >>> # Initialize configuration and logger
    >>> config = get_config()
    >>> logger = get_logger(__name__)
    >>>
    >>> # Test API availability
    >>> tester = APITester()
    >>> results = tester.run_test()
"""

# Core utilities
from codebase._utils import (
    Config,
    get_config,
    get_logger,
    FeishuBot,
    get_feishu_bot,
    send_feishu_message,
)

# API testing
from codebase.test import (
    APITester,
    StreamType,
    APIConfig,
)

# LLM operations
from codebase.llm import LLMClient

# Data pipelines
from codebase.llm_pipeline import LLMPipeline

__version__ = "0.1.0"

__all__ = [
    # Core utilities
    "Config",
    "get_config",
    "get_logger",
    "FeishuBot",
    "get_feishu_bot",
    "send_feishu_message",
    # API testing
    "APITester",
    "StreamType",
    "APIConfig",
    # LLM operations
    "LLMClient",
    # Data pipelines
    "LLMPipeline",
]
