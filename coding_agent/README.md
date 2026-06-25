# Coding Agent

一个从零手写的 Python Coding Agent，不依赖任何 Agent 框架（LangChain 等），使用 OpenAI 兼容 API。

## 功能

- 🤖 终端交互式对话，Agent 可以主动调用工具完成任务
- 📖 **read_file** — 读取文件内容，带行号，支持行号范围
- 📂 **list_directory** — 列出目录内容，标注文件和目录
- 🔍 **search_in_file** — 在文件中搜索关键词，带上下文
- ⚙️ **execute_command** — 执行安全的 shell 命令（三级安全分级）
- ✏️ **write_file** — 创建或修改文件（支持覆盖/追加，路径沙盒保护）
- ❓ **ask_followup_question** — 不确定时主动向用户澄清
- 📦 智能上下文压缩（三级策略：轻量 / 中度滑动窗口 / 重度 LLM 摘要）
- ⚡ 流式输出（实时逐字显示文本响应）
- 🔄 API 重试（指数退避，最多 3 次）
- 💾 会话持久化（SQLite，自动存档 + 手动命名保存 / 恢复）
- 📊 实时显示 token 用量
- 🛡️ 结构化异常体系（工具错误有明确类型）

## 环境要求

- Python 3.11+
- 一个兼容 OpenAI API 格式的服务（DeepSeek、OpenAI 等）

## 安装

```bash
# 1. 进入项目目录
cd coding_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API_KEY

# 4. 运行
python agent.py
```

## 配置说明

编辑 `.env` 文件：

```ini
# --- API 配置（必填）---
API_KEY=sk-your-api-key-here

# 模型名称
MODEL_NAME=deepseek-chat

# API 地址
BASE_URL=https://api.deepseek.com

# --- 模型参数 ---
TEMPERATURE=0.7          # 温度（0-2）
TOP_P=1.0                # Top-P 采样

# --- 流式输出 ---
STREAMING_ENABLED=true   # 开启后文本响应实时逐字显示

# --- API 重试 ---
MAX_RETRIES=3            # 最大重试次数
RETRY_BASE_DELAY=1.0     # 重试基础延迟（秒）

# --- 会话管理 ---
MAX_TOOL_ROUNDS=5        # 最大工具调用轮数
MAX_RECENT_ROUNDS=10     # 上下文窗口保留轮数
CONTEXT_LIMIT_TOKENS=80000  # 上下文 token 上限
```

## 特殊命令

| 命令 | 作用 |
|------|------|
| `/exit` | 退出程序（自动保存会话） |
| `/clear` | 清空对话上下文 |
| `/help` | 显示帮助信息 |
| `/stats` | 查看当前会话统计 |
| `/save <名称>` | 保存当前会话为命名存档 |
| `/load <名称>` | 加载命名存档 |
| `/sessions` | 列出所有已保存的会话 |

## 安全说明

`execute_command` 工具实现了三级安全机制：

1. **安全命令**（直接执行）：`ls`、`cat`、`grep`、`find`、`git`、`python`、`pytest` 等
2. **需确认命令**（询问用户后执行）：`pip install`、`npm install`、`curl`、`mv`、`cp` 等
3. **禁止命令**（直接拒绝）：`rm`、`sudo`、`chmod 777`、fork bomb、管道到 shell 等

此外，简单命令优先使用 `shell=False` 执行以增强安全性。

## 项目结构

```
coding_agent/
├── .env.example        # 环境变量模板
├── .env                # 你的私有配置（不提交到 git）
├── agent.py            # 主入口，Agent 核心循环
├── tools.py            # 工具函数 + 工具注册 + JSON Schema
├── exceptions.py       # 异常体系定义
├── session_store.py    # 会话持久化（SQLite）
├── requirements.txt    # 依赖声明
├── pyproject.toml      # 项目元数据
└── README.md           # 本文件
```

## 扩展指南

要新增一个工具，只需在 `tools.py` 中做三件事：

```python
# ① 写函数（错误用异常抛出）
def my_tool(param1: str) -> str:
    """工具描述"""
    if not param1:
        raise ToolInputError("param1 is required", tool_name="my_tool")
    return f"结果: {param1}"

# ② 注册映射
AVAILABLE_TOOLS["my_tool"] = my_tool

# ③ 添加 JSON Schema
TOOLS.append({...})
```
