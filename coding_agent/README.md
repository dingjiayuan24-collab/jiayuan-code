# Coding Agent

一个从零手写的 Python Coding Agent，不依赖任何 Agent 框架（LangChain 等），使用 OpenAI 兼容 API。

## 功能

- 🤖 终端交互式对话，Agent 可以主动调用工具完成任务
- 📖 **read_file** — 读取文件内容，带行号，支持行号范围
- 📂 **list_directory** — 列出目录内容，标注文件和目录
- 🔍 **search_in_file** — 在文件中搜索关键词，带上下文
- ⚙️ **execute_command** — 执行安全的 shell 命令（有安全白名单）
- 📦 自动上下文压缩（滑动窗口），防止 token 超限
- 📊 实时显示 token 用量

## 环境要求

- Python 3.11+
- 一个兼容 OpenAI API 格式的服务（DeepSeek、OpenAI 等）

## 安装

```bash
# 1. 进入项目目录
cd coding_agent

# 2. 安装依赖
pip install openai python-dotenv tiktoken

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API_KEY

# 4. 运行
python agent.py
```

## 配置说明

编辑 `.env` 文件：

```ini
# 必填：你的 API Key
API_KEY=sk-your-api-key-here

# 模型名称（默认 deepseek-chat，也可以用 gpt-4o、claude 兼容接口等）
MODEL_NAME=deepseek-chat

# API 地址（DeepSeek 默认地址，换成 OpenAI 就是 https://api.openai.com/v1）
BASE_URL=https://api.deepseek.com

# 最大工具调用轮数（一次用户输入最多让 Agent 调用多少次工具）
MAX_TOOL_ROUNDS=5

# 上下文窗口保留的最近对话轮数
MAX_RECENT_ROUNDS=10
```

## 使用示例

```
🤖 Coding Agent 启动
   模型: deepseek-chat
   地址: https://api.deepseek.com
   最大工具轮数: 5
   上下文窗口: 保留最近 10 轮对话

💬 开始对话（输入 /exit 退出，/clear 清空上下文）
───────────────────────────────────────────────────────

👤 你: 帮我看看当前目录下有什么文件

  🧠 [思考中] 轮次 1...
  🔧 [调用] list_directory({"dir_path": "."})
  📋 [结果] 245 字符
  📊 [用量] 输入 620 token | 输出 45 token | 总计 665 token

  🧠 [思考中] 轮次 2...

🤖 Agent:
当前目录下有这些文件：
- agent.py
- tools.py
- .env
- README.md

👤 你: 读一下 tools.py 的前 30 行

  🧠 [思考中] 轮次 1...
  🔧 [调用] read_file({"file_path": "tools.py", "line_start": 1, "line_end": 30})
  ...

👤 你: /exit
👋 再见！
```

## 特殊命令

| 命令 | 作用 |
|------|------|
| `/exit` | 退出程序 |
| `/clear` | 清空对话上下文 |

## 安全说明

`execute_command` 工具实现了两层安全机制：

1. **命令白名单**：只允许 `ls`、`cat`、`grep`、`find`、`git`、`python`、`pytest` 等开发常用命令
2. **危险模式检测**：禁止包含 `rm`、`sudo`、`chmod 777`、fork bomb、管道到 shell 等危险操作

白名单以外的命令会被拒绝执行。

## 扩展指南

要新增一个工具，只需在 `tools.py` 中做三件事：

```python
# ① 写函数
def my_tool(param1: str) -> str:
    """工具描述"""
    return f"结果: {param1}"

# ② 注册映射
AVAILABLE_TOOLS["my_tool"] = my_tool

# ③ 添加 JSON Schema
TOOLS.append({
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."}
            },
            "required": ["param1"],
        },
    },
})
```

## 项目结构

```
coding_agent/
├── .env.example   # 环境变量模板
├── .env           # 你的私有配置（不提交到 git）
├── agent.py       # 主入口，Agent 核心循环
├── tools.py       # 工具函数 + 工具注册 + JSON Schema
└── README.md      # 本文件
```
