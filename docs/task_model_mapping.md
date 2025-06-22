# Task-to-Model Mapping Configuration

This document explains how to customize which models are used for different types of tasks in the Zen MCP Server.

## Overview

The Zen MCP Server automatically selects appropriate models based on the type of task being performed. You can customize these selections using a simple JSON configuration file.

## Task Categories

### **Extended Reasoning**
Tools that require deep analysis and complex reasoning:
- `thinkdeep` - Deep analysis and reasoning
- `debug` - Complex debugging and root cause analysis  
- `codereview` - Thorough code analysis
- `refactor` - Complex refactoring analysis
- `docgen` - Comprehensive documentation generation
- `secaudit` - Security analysis

### **Fast Response**  
Tools that prioritize speed and cost efficiency:
- `chat` - General conversation and quick questions

### **Balanced**
Tools that need balanced performance and capability:
- `analyze` - File and code analysis
- `consensus` - Multi-model consensus building
- `planner` - Project planning  
- `testgen` - Test generation
- `tracer` - Code tracing

## Configuration File

Create or edit `conf/task_model_mapping.json`:

```json
{
  "enabled": true,
  "mappings": {
    "extended_reasoning": {
      "preferred_models": ["opus", "pro", "deepseek-r1", "sonnet"]
    },
    "fast_response": {
      "preferred_models": ["flash", "haiku", "sonnet"]  
    },
    "balanced": {
      "preferred_models": ["sonnet", "flash", "pro"]
    }
  },
  "tool_overrides": {
    "enabled": true,
    "overrides": {
      "chat": {
        "preferred_models": ["flash", "haiku"]
      },
      "thinkdeep": {
        "preferred_models": ["opus", "deepseek-r1", "pro"]
      }
    }
  }
}
```

## Model Names

You can use either aliases from `custom_models.json` or full model names:

**OpenRouter Aliases:**
- `opus` → `anthropic/claude-3-opus`
- `sonnet` → `anthropic/claude-3-sonnet`  
- `haiku` → `anthropic/claude-3-haiku`
- `pro` → `google/gemini-2.5-pro`
- `flash` → `google/gemini-2.5-flash`
- `deepseek-r1` → `deepseek/deepseek-r1-0528`

**Or Full Names:**
- `anthropic/claude-3-opus`
- `google/gemini-2.5-pro`
- `mistralai/mistral-large-2411`

## How It Works

1. **Tool Execution**: When a tool runs in auto mode (`DEFAULT_MODEL=auto`)
2. **Category Check**: System determines the tool's category  
3. **Configuration Lookup**: Checks your custom configuration
4. **Model Selection**: Uses first available model from your preferred list
5. **Fallback**: If no preferred models available, uses built-in defaults

## Priority Order

1. **Tool-specific overrides** (highest priority)
2. **Category-based mappings**  
3. **Built-in defaults** (fallback)

## Examples

### Cost Optimization Setup
```json
{
  "enabled": true,
  "mappings": {
    "extended_reasoning": {
      "preferred_models": ["sonnet", "flash"]
    },
    "fast_response": {
      "preferred_models": ["flash", "haiku"]
    },
    "balanced": {
      "preferred_models": ["flash", "sonnet"]
    }
  }
}
```

### Performance Optimization Setup  
```json
{
  "enabled": true,
  "mappings": {
    "extended_reasoning": {
      "preferred_models": ["opus", "pro", "deepseek-r1"]
    },
    "fast_response": {
      "preferred_models": ["flash", "sonnet"]
    },
    "balanced": {
      "preferred_models": ["sonnet", "pro"]
    }
  }
}
```

### Single Model Setup
```json
{
  "enabled": true,
  "mappings": {
    "extended_reasoning": {
      "preferred_models": ["sonnet"]
    },
    "fast_response": {
      "preferred_models": ["sonnet"]  
    },
    "balanced": {
      "preferred_models": ["sonnet"]
    }
  }
}
```

## Environment Variable Override

You can also specify the config file location:

```bash
export TASK_MODEL_CONFIG_PATH=/path/to/your/custom_mapping.json
```

## Disabling Custom Mappings

Set `"enabled": false` in the configuration file to use built-in defaults.

## Debugging

The server logs will show when custom models are selected:

```
INFO - Using configured preferred model 'opus' for extended_reasoning task
```