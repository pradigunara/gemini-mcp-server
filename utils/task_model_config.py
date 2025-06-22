"""
Task-to-Model Mapping Configuration System

This module provides easy configuration of model selection for different tool categories.
Users can customize which models are preferred for extended reasoning, fast response,
and balanced tasks through a simple JSON configuration file.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from utils.file_utils import read_json_file

logger = logging.getLogger(__name__)


class TaskModelConfig:
    """Configuration manager for task-to-model mappings."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize task model configuration.
        
        Args:
            config_path: Path to config file. If None, uses default location.
        """
        self.config_path = self._get_config_path(config_path)
        self.config = self._load_config()
        
    def _get_config_path(self, config_path: Optional[str]) -> Path:
        """Get configuration file path."""
        if config_path:
            return Path(config_path)
        
        # Check environment variable
        env_path = os.getenv("TASK_MODEL_CONFIG_PATH")
        if env_path:
            return Path(env_path)
            
        # Default to conf/task_model_mapping.json
        return Path(__file__).parent.parent / "conf" / "task_model_mapping.json"
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if not self.config_path.exists():
            logger.debug(f"Task model config not found at {self.config_path}, using defaults")
            return {"enabled": False}
            
        try:
            config = read_json_file(str(self.config_path))
            if config is None:
                logger.warning(f"Could not read task model config from {self.config_path}")
                return {"enabled": False}
                
            logger.debug(f"Loaded task model configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading task model config: {e}")
            return {"enabled": False}
    
    def is_enabled(self) -> bool:
        """Check if custom task model mapping is enabled."""
        return self.config.get("enabled", False)
    
    def get_preferred_models_for_category(self, category: str) -> List[str]:
        """Get preferred model list for a tool category.
        
        Args:
            category: Tool category ('extended_reasoning', 'fast_response', 'balanced')
            
        Returns:
            List of preferred model names in order of preference
        """
        if not self.is_enabled():
            return []
            
        mappings = self.config.get("mappings", {})
        category_config = mappings.get(category, {})
        return category_config.get("preferred_models", [])
    
    def get_tool_override(self, tool_name: str) -> Optional[Dict]:
        """Get tool-specific override configuration.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Override configuration or None if no override exists
        """
        if not self.is_enabled():
            return None
            
        overrides = self.config.get("tool_overrides", {})
        if not overrides.get("enabled", False):
            return None
            
        return overrides.get("overrides", {}).get(tool_name)
    
    def get_effective_category_for_tool(self, tool_name: str, default_category: str) -> str:
        """Get effective category for a tool, considering overrides.
        
        Args:
            tool_name: Name of the tool
            default_category: Default category from tool implementation
            
        Returns:
            Effective category to use for model selection
        """
        override = self.get_tool_override(tool_name)
        if override and "category" in override:
            return override["category"]
        return default_category
    
    def get_effective_models_for_tool(self, tool_name: str, category: str) -> List[str]:
        """Get effective preferred models for a tool.
        
        Args:
            tool_name: Name of the tool  
            category: Tool category
            
        Returns:
            List of preferred models, considering tool overrides
        """
        # Check for tool-specific model override first
        override = self.get_tool_override(tool_name)
        if override and "preferred_models" in override:
            return override["preferred_models"]
            
        # Fall back to category-based mapping
        return self.get_preferred_models_for_category(category)


# Global instance
_task_model_config = None


def get_task_model_config() -> TaskModelConfig:
    """Get global task model configuration instance."""
    global _task_model_config
    if _task_model_config is None:
        _task_model_config = TaskModelConfig()
    return _task_model_config


def reload_task_model_config():
    """Reload task model configuration from disk."""
    global _task_model_config
    _task_model_config = None
    return get_task_model_config()