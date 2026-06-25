"""
agent.py —— Coding Agent 主入口
================================
这个文件实现了 Agent 的核心循环逻辑：
  1. 加载配置（.env）
  2. 初始化 OpenAI 兼容客户端
  3. 项目自动感知（git 状态、语言检测、结构摘要）
  4. 维护对话历史 + 智能上下文压缩（三级策略）
  5. 调用 LLM API（支持流式输出 + 指数退避重试）

v2 变更：
  - 支持流式输出（STREAMING），文本响应实时打印
  - API 调用增加指数退避重试（最多 3 次）
  - execute_tool_call 捕获 ToolError 异常，格式化后返回给模型
  - model 参数（temperature, top_p）可通过 .env 配置
  - 会话持久化（SQLite），支持保存/恢复/自动存档

运行方式：
  python agent.py

依赖安装：
  pip install -r requirements.txt
"""

import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# 导入自定义模块
from exceptions import AgentAPIError, ToolError, ToolInputError, ToolSecurityError, ToolTimeoutError
from session_store import (
    auto_save,
    delete_session,
    has_auto_saved_session,
    list_saved_sessions,
    load_session,
    save_session,
)
from tools import (
    AVAILABLE_TOOLS,
    TOOLS,
    clear_confirm_cache,
    set_workspace_root,
)

# ============================================================
# 全局 Rich Console 实例
# ============================================================

console = Console()

# ============================================================
# 0. 加载环境变量
# ============================================================

load_dotenv(override=True)

API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))
MAX_RECENT_ROUNDS = int(os.getenv("MAX_RECENT_ROUNDS", "10"))
CONTEXT_LIMIT_TOKENS = int(os.getenv("CONTEXT_LIMIT_TOKENS", "80000"))

# --- 模型参数（可通过 .env 自定义）---
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "1.0"))

# --- 流式输出开关 ---
STREAMING_ENABLED = os.getenv("STREAMING_ENABLED", "true").lower() in ("true", "1", "yes")

# --- API 重试参数 ---
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))

if not API_KEY:
    console.print(Panel(
        "[bold red]未设置 API_KEY[/bold red]\n"
        "请在 .env 文件中配置，可以参考 .env.example 创建 .env 文件",
        title="❌ 配置错误",
        border_style="red",
    ))
    sys.exit(1)

# 设置工具系统的工作区根目录
WORKSPACE_ROOT = os.getcwd()
set_workspace_root(WORKSPACE_ROOT)

# ============================================================
# 1. 初始化 OpenAI 兼容客户端
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)


# ============================================================
# 2. 启动画面
# ============================================================

def print_banner():
    """打印 Agent 启动时的欢迎面板。"""
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("key", style="bold cyan", width=14)
    info_table.add_column("value", style="white")
    info_table.add_row("模型", MODEL_NAME)
    info_table.add_row("地址", BASE_URL)
    info_table.add_row("工作目录", WORKSPACE_ROOT)
    info_table.add_row("工具数量", str(len(TOOLS)))
    info_table.add_row("最大工具轮数", str(MAX_TOOL_ROUNDS))
    info_table.add_row("上下文上限", f"{CONTEXT_LIMIT_TOKENS:,} tokens")
    info_table.add_row("流式输出", "开启" if STREAMING_ENABLED else "关闭")
    info_table.add_row("Temperature", str(TEMPERATURE))
    info_table.add_row("Top-P", str(TOP_P))

    console.print()
    console.print(Panel(
        info_table,
        title="[bold white]🤖 Coding Agent[/bold white]",
        subtitle="[dim]输入 /exit 退出 | /clear 清空上下文 | /help 查看帮助[/dim]",
        border_style="cyan",
        padding=(1, 2),
    ))


def print_goodbye():
    """打印退出时的告别面板。"""
    console.print()
    console.print(Panel(
        "[dim]感谢使用，再见！[/dim]",
        title="[bold]👋 再见[/bold]",
        border_style="cyan",
    ))


def print_help():
    """打印帮助面板。"""
    help_text = """
[bold]可用命令:[/bold]

  /exit      退出程序（自动保存会话）
  /clear     清空对话上下文
  /help      显示此帮助信息
  /stats     查看当前会话统计
  /save      将当前会话保存为一个命名存档
  /load      加载一个命名存档（替换当前会话）
  /sessions  列出所有已保存的会话

[bold]可用工具（由模型自动调用）:[/bold]

  read_file             读取文件内容，带行号
  list_directory        列出目录内容
  search_in_file        在文件中搜索关键词
  execute_command       执行安全的 shell 命令
  write_file            创建或修改文件
  ask_followup_question 向用户澄清不确定的问题

[bold]配置:[/bold]
  编辑 .env 文件来修改模型、API 地址等参数。
  详细说明见 README.md
"""
    console.print(Panel(help_text, title="[bold white]📖 帮助[/bold white]", border_style="cyan"))


def print_stats(messages: list[dict], start_time: float) -> None:
    """打印当前会话统计。"""
    elapsed = time.time() - start_time
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    est_tokens = estimate_tokens(messages)

    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("key", style="bold cyan", width=16)
    stats_table.add_column("value", style="white")
    stats_table.add_row("运行时长", f"{elapsed:.0f} 秒")
    stats_table.add_row("用户消息", str(len(user_msgs)))
    stats_table.add_row("助手消息", str(len(assistant_msgs)))
    stats_table.add_row("工具调用", str(len(tool_msgs)))
    stats_table.add_row("总消息数", str(len(messages)))
    stats_table.add_row("估算 token", f"~{est_tokens:,}")

    console.print()
    console.print(Panel(stats_table, title="[bold white]📊 会话统计[/bold white]", border_style="cyan"))


def print_sessions() -> None:
    """打印所有已保存的会话列表。"""
    sessions = list_saved_sessions()

    if not sessions:
        console.print("  [dim]没有已保存的会话[/dim]")
        return

    table = Table(title="已保存的会话", border_style="cyan")
    table.add_column("名称", style="bold cyan")
    table.add_column("消息数", justify="right")
    table.add_column("创建时间")
    table.add_column("更新时间")

    for s in sessions:
        table.add_row(
            s["name"].replace("__auto__", "(自动保存)"),
            str(s["message_count"]),
            s["created_at"],
            s["updated_at"],
        )

    console.print()
    console.print(table)


# ============================================================
# 3. 项目感知 —— 启动时自动检测项目信息
# ============================================================

def detect_project() -> str:
    """
    自动检测当前项目的信息，返回一段描述字符串。

    检测内容：
      1. Git 仓库状态（分支、最近提交）
      2. 项目主要编程语言
      3. 项目根目录结构摘要
    """
    import subprocess as sp

    parts = []

    # --- 3.1 Git 仓库检测 ---
    try:
        git_dir = sp.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True, text=True, timeout=5, cwd=WORKSPACE_ROOT
        )
        if git_dir.returncode == 0:
            branch = sp.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=WORKSPACE_ROOT
            )
            branch_name = branch.stdout.strip() or "unknown"

            log = sp.run(
                ["git", "log", "--oneline", "-3", "--format=%s"],
                capture_output=True, text=True, timeout=5, cwd=WORKSPACE_ROOT
            )
            commits = [c.strip() for c in log.stdout.strip().split("\n") if c.strip()]

            parts.append(f"Git 分支: {branch_name}")
            if commits:
                parts.append(f"最近提交: {' | '.join(commits[:3])}")
    except Exception:
        pass

    # --- 3.2 项目语言检测 ---
    try:
        ext_counter = Counter()
        flag_files = {
            "Cargo.toml": "Rust",
            "go.mod": "Go",
            "package.json": "JavaScript/TypeScript",
            "tsconfig.json": "TypeScript",
            "CMakeLists.txt": "C/C++ (CMake)",
            "Makefile": "C/C++ (Make)",
            "pom.xml": "Java (Maven)",
            "build.gradle": "Java (Gradle)",
            "Gemfile": "Ruby",
            "mix.exs": "Elixir",
            "Cargo.lock": "Rust",
        }

        detected_flags = []
        root_entries = list(Path(WORKSPACE_ROOT).iterdir())
        for entry in root_entries:
            if entry.name in flag_files:
                detected_flags.append(flag_files[entry.name])

            if entry.is_file():
                ext = entry.suffix.lower()
                if ext:
                    ext_counter[ext] += 1

        for entry in root_entries:
            if entry.is_dir() and not entry.name.startswith(".") and entry.name not in (
                "__pycache__", "node_modules", "target", "venv", ".venv", "dist", "build"
            ):
                try:
                    for sub_entry in entry.iterdir():
                        if sub_entry.is_file():
                            ext = sub_entry.suffix.lower()
                            if ext:
                                ext_counter[ext] += 1
                except PermissionError:
                    pass

        lang_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".tsx": "TypeScript (React)",
            ".jsx": "JavaScript (React)",
            ".rs": "Rust",
            ".go": "Go",
            ".java": "Java",
            ".c": "C",
            ".cpp": "C++",
            ".h": "C/C++ Header",
            ".rb": "Ruby",
            ".ex": "Elixir",
            ".exs": "Elixir",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".scala": "Scala",
            ".vue": "Vue.js",
            ".svelte": "Svelte",
            ".css": "CSS",
            ".html": "HTML",
            ".md": "Markdown",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".toml": "TOML",
        }

        if detected_flags:
            parts.append(f"项目类型: {', '.join(detected_flags)}")
        elif ext_counter:
            top_ext = ext_counter.most_common(3)
            top_langs = []
            for ext, count in top_ext:
                lang = lang_map.get(ext, ext)
                top_langs.append(f"{lang}({count})")
            parts.append(f"项目语言: {', '.join(top_langs)}")
    except Exception:
        pass

    # --- 3.3 项目结构摘要 ---
    try:
        noise_dirs = {
            ".git", "__pycache__", "node_modules", "target",
            "venv", ".venv", "dist", "build", ".idea", ".vscode",
            ".claude", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        }
        structure_lines = []
        root_entries = sorted(
            [e for e in Path(WORKSPACE_ROOT).iterdir() if e.name not in noise_dirs],
            key=lambda e: (not e.is_dir(), e.name.lower())
        )
        for entry in root_entries[:30]:
            if entry.is_dir():
                structure_lines.append(f"  📁 {entry.name}/")
            else:
                size = entry.stat().st_size
                structure_lines.append(f"  📄 {entry.name} ({_format_size_simple(size)})")

        if structure_lines:
            parts.append(f"项目结构:\n" + "\n".join(structure_lines))
    except Exception:
        pass

    if not parts:
        return "（未检测到特殊项目信息）"

    return "\n".join(parts)


def _format_size_simple(size_bytes: int) -> str:
    """简易版文件大小格式化（用于项目结构展示）。"""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes}{unit}" if unit == "B" else f"{size_bytes:.0f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}GB"


# ============================================================
# 4. 系统提示词 —— 动态拼接项目信息
# ============================================================

def build_system_prompt(project_info: str) -> str:
    """根据项目检测结果构建系统提示词。"""
    return f"""你是一个 Coding Agent，运行在用户的终端中，帮助用户完成编程相关的任务。

## 当前环境
- 工作目录: {WORKSPACE_ROOT}
- 操作系统: {sys.platform}
- 日期: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## 项目信息
{project_info}

## 你的能力
- 使用 read_file 工具读取文件内容（支持指定行号范围）
- 使用 list_directory 工具浏览目录结构
- 使用 search_in_file 工具在文件中搜索关键词
- 使用 execute_command 工具执行安全的 shell 命令（有安全分级）
- 使用 write_file 工具创建或修改文件（支持覆盖和追加模式）
- 使用 ask_followup_question 工具向用户澄清不确定的问题

## 工作原则
1. **主动使用工具**：遇到需要查看文件、搜索代码、执行命令的场景，直接调用工具，不要凭空猜测。
2. **不确定时提问**：对用户的需求有疑问时，使用 ask_followup_question 工具问清楚，不要自己猜。
3. **逐步推进**：一次只做一件事，根据上一步的结果决定下一步。
4. **简洁回复**：给出关键信息即可，不需要长篇大论。
5. **诚实透明**：工具返回什么就如实汇报，不要编造内容。
6. **安全第一**：写文件前确认路径安全，执行命令前注意安全分级。"""


# ============================================================
# 5. Token 估算
# ============================================================

def estimate_tokens(messages: list[dict]) -> int:
    """
    估算消息列表的总 token 数。

    优先使用 tiktoken（cl100k_base 编码器，GPT-4 / DeepSeek 通用），
    如果 tiktoken 未安装则使用字符数 / 2 的粗略估算。

    估算结果用于判断是否需要压缩，不需要精确到个位。
    """
    if _TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            total = 0
            for msg in messages:
                total += 4
                for key, value in msg.items():
                    if isinstance(value, str):
                        total += len(enc.encode(value))
                    elif isinstance(value, list):
                        total += len(enc.encode(json.dumps(value, ensure_ascii=False)))
                total += 2
            total += 2
            return total
        except Exception:
            pass  # fall through to fallback

    # fallback: 粗略估算（中英文混合约 2 字符 / token）
    total_chars = sum(len(str(m)) for m in messages)
    return total_chars // 2


# ============================================================
# 6. 智能上下文压缩 —— 三级策略
# ============================================================

def _summarize_with_llm(messages_to_summarize: list[dict]) -> str:
    """
    调用模型本身对历史消息做摘要。
    使用 temperature=0 追求稳定输出。
    带重试机制。
    """
    summary_prompt = (
        "请用中文总结以下对话历史，包含四个部分：\n"
        "1. 用户曾要求做什么\n"
        "2. 已完成的操作\n"
        "3. 涉及的关键文件\n"
        "4. 未解决的问题\n\n"
        "保持简洁，每部分用一句话概括。"
    )

    summary_messages = messages_to_summarize + [
        {"role": "user", "content": summary_prompt}
    ]

    try:
        response = call_api_with_retry(
            model=MODEL_NAME,
            messages=summary_messages,
            temperature=0,
            max_tokens=500,
        )
        return response.choices[0].message.content or "（摘要生成失败）"
    except Exception as e:
        return f"（摘要生成失败: {e}）"


def compress_messages(messages: list[dict]) -> list[dict]:
    """
    智能上下文压缩 —— 三级策略：

    1. 轻量（token < 70% 上限）：不压缩，原样返回
    2. 中度（70%-90%）：保留 system + 最近 4 轮完整 + 更早轮次截断 assistant
    3. 重度（> 90%）：保留 system + 最近 2 轮 + 其余由模型摘要

    参数:
        messages: 当前完整消息列表

    返回:
        压缩后的消息列表（可能是原样）
    """
    if not messages:
        return messages

    current_tokens = estimate_tokens(messages)

    # --- 轻量压缩：不触发 ---
    light_threshold = int(CONTEXT_LIMIT_TOKENS * 0.70)
    if current_tokens < light_threshold:
        return messages

    # 分离 system 消息
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    # 找到 user 消息的索引位置
    user_indices = [i for i, m in enumerate(other_msgs) if m["role"] == "user"]

    medium_threshold = int(CONTEXT_LIMIT_TOKENS * 0.90)

    # --- 中度压缩：保留最近 4 轮，截断更早的 assistant 消息 ---
    if current_tokens < medium_threshold:
        keep_rounds = 4
        if len(user_indices) <= keep_rounds:
            return messages

        cut_idx = user_indices[-(keep_rounds)]
        recent_msgs = other_msgs[cut_idx:]
        older_msgs = other_msgs[:cut_idx]

        truncated = []
        for msg in older_msgs:
            if msg["role"] == "user":
                truncated.append(msg)
            elif msg["role"] == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 500:
                    new_msg = msg.copy()
                    new_msg["content"] = content[:200] + "\n...(已截断)"
                    truncated.append(new_msg)
                else:
                    truncated.append(msg)
            elif msg["role"] == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 200:
                    new_msg = msg.copy()
                    new_msg["content"] = content[:200] + "\n...(已截断)"
                    truncated.append(new_msg)
                else:
                    truncated.append(msg)

        summary_hint = {
            "role": "system",
            "content": "[以下是更早的对话摘要，部分 assistant 回复已被截断]",
        }

        compressed = system_msgs + [summary_hint] + truncated + recent_msgs

        after_tokens = estimate_tokens(compressed)
        console.print(
            f"  [yellow]📦 中度压缩[/yellow]: "
            f"[dim]{len(messages)} 条 → {len(compressed)} 条 | "
            f"~{current_tokens} → ~{after_tokens} tokens[/dim]"
        )
        return compressed

    # --- 重度压缩：保留最近 2 轮，其余由模型摘要 ---
    keep_rounds = 2
    if len(user_indices) <= keep_rounds:
        pass

    cut_idx = user_indices[-(keep_rounds)] if len(user_indices) >= keep_rounds else 0
    recent_msgs = other_msgs[cut_idx:]
    older_msgs = other_msgs[:cut_idx]

    if older_msgs:
        console.print("  [yellow]📦 重度压缩[/yellow]: [dim]正在调用模型生成摘要...[/dim]")
        summary = _summarize_with_llm(older_msgs)
        summary_msg = {
            "role": "system",
            "content": f"[历史摘要] {summary}",
        }
    else:
        summary_msg = None

    compressed = system_msgs[:]
    if summary_msg:
        compressed.append(summary_msg)
    compressed.extend(recent_msgs)

    after_tokens = estimate_tokens(compressed)
    console.print(
        f"  [yellow]📦 重度压缩[/yellow]: "
        f"[dim]{len(messages)} 条 → {len(compressed)} 条 | "
        f"~{current_tokens} → ~{after_tokens} tokens[/dim]"
    )
    return compressed


# ============================================================
# 7. API 调用（带指数退避重试）
# ============================================================

def call_api_with_retry(**create_kwargs) -> any:
    """
    封装 OpenAI API 调用，失败时自动重试（指数退避）。

    重试策略:
      - 第 1 次重试：等待 base_delay 秒
      - 第 2 次重试：等待 base_delay * 2 秒
      - 第 3 次重试：等待 base_delay * 4 秒（最后一次）
      - 每次加上随机抖动（0~1 秒）避免惊群效应

    参数:
        **create_kwargs: 传给 client.chat.completions.create() 的所有参数

    返回:
        API 响应对象

    异常:
        AgentAPIError: 所有重试都失败后抛出
    """
    last_error = None
    max_attempts = MAX_RETRIES

    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**create_kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                console.print(
                    f"  [yellow]⚠️ API 调用失败 (尝试 {attempt + 1}/{max_attempts})"
                    f"—— {delay:.1f}s 后重试: {e}[/yellow]"
                )
                time.sleep(delay)
            else:
                console.print(
                    f"  [red]❌ API 调用失败，已重试 {max_attempts} 次: {e}[/red]"
                )

    raise AgentAPIError(
        f"API 调用失败，已重试 {max_attempts} 次: {last_error}"
    )


# ============================================================
# 8. 执行单个工具调用（异常感知 + rich 日志）
# ============================================================

def execute_tool_call(tool_call: dict) -> str:
    """
    根据 API 返回的 tool_call 字典，找到对应的函数并执行。
    捕获 ToolError 异常并格式化为给模型看的错误消息。

    参数:
        tool_call: tool_call.model_dump() 的结果

    返回:
        工具执行结果字符串（成功结果或格式化后的错误消息）
    """
    func_name = tool_call["function"]["name"]

    # --- 检查工具是否存在 ---
    if func_name not in AVAILABLE_TOOLS:
        return f"错误：未知工具 —— {func_name}"

    # --- 解析 JSON 参数 ---
    try:
        arguments = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError as e:
        return f"错误：无法解析工具参数 JSON —— {e}"

    # --- 日志：蓝色粗体函数名 + dim 参数 ---
    args_str = json.dumps(arguments, ensure_ascii=False)
    if len(args_str) > 200:
        args_str = args_str[:200] + "..."

    console.print(
        f"  [bold blue]🔧 {func_name}[/bold blue]"
        f"[dim]({args_str})[/dim]"
    )

    # --- 调用工具函数，捕获 ToolError 异常 ---
    func = AVAILABLE_TOOLS[func_name]

    try:
        result = func(**arguments)
    except ToolInputError as e:
        result = f"参数错误: {e}"
    except ToolSecurityError as e:
        result = f"⛔ 安全限制: {e}"
        console.print(f"  [bold red]⛔ 安全限制触发[/bold red]")
    except ToolTimeoutError as e:
        result = f"⏱ 超时: {e}"
        console.print(f"  [bold red]⏱ 超时[/bold red]")
    except ToolError as e:
        result = f"错误: {e}"
        console.print(f"  [bold red]❌ 工具错误: {e}[/bold red]")
    except TypeError as e:
        result = f"错误：参数不匹配 —— {e}，收到参数: {arguments}"
        console.print(f"  [bold red]❌ 参数不匹配: {e}[/bold red]")
    except Exception as e:
        result = f"错误：工具执行异常 —— {e}"
        console.print(f"  [bold red]❌ 未知异常: {e}[/bold red]")

    # --- 截断过长的结果 ---
    max_result_len = 8000
    if len(result) > max_result_len:
        result = result[:max_result_len] + (
            f"\n\n...(结果过长，已截断，原长度 {len(result)} 字符)"
        )

    # --- 结果日志：根据内容用不同颜色 ---
    if result.startswith("错误") or result.startswith("⛔") or result.startswith("参数错误"):
        color = "red"
    elif result.startswith("⚠") or result.startswith("⏭") or result.startswith("⏱"):
        color = "yellow"
    else:
        color = "green"

    console.print(f"  [{color}]📋 返回 {len(result)} 字符[/{color}]")

    return result


# ============================================================
# 9. 流式 API 调用 + 处理
# ============================================================

def _process_streaming_response(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """
    使用流式模式调用 API 并处理响应。

    返回:
        (text_content, tool_calls_list)
        - 如果是文本响应: text_content 有值, tool_calls_list 为空列表
        - 如果是工具调用: text_content 为 None, tool_calls_list 有值
        - 如果两者都无: 返回 (None, [])，由调用方处理
    """
    stream = call_api_with_retry(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=4096,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stream=True,
    )

    content_buffer = ""
    tool_calls_acc: dict[int, dict] = {}  # index -> {id, function: {name, arguments}}
    usage_info = None
    stream_error = None

    try:
        for chunk in stream:
            # 收集 usage
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 处理文本内容：实时逐字打印
            if delta.content:
                content_buffer += delta.content
                console.print(delta.content, end="", highlight=False)
                sys.stdout.flush()  # 确保即时输出

            # 处理工具调用：累积 delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_acc[idx]
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            entry["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            entry["function"]["arguments"] += tc.function.arguments
    except Exception as e:
        # 流式传输中断（如网络问题），记录错误
        stream_error = str(e)
    finally:
        # 确保流被正确关闭
        if hasattr(stream, "close"):
            try:
                stream.close()
            except Exception:
                pass

    # 打印 usage
    if usage_info:
        console.print(
            f"\n  [dim]📊 输入 {usage_info.prompt_tokens} | "
            f"输出 {usage_info.completion_tokens} | "
            f"总计 {usage_info.total_tokens} tokens[/dim]"
        )

    # 流式传输出错
    if stream_error and not content_buffer and not tool_calls_acc:
        console.print(f"\n  [red]❌ 流式传输中断: {stream_error}[/red]")
        return None, []

    if content_buffer:
        console.print()  # stream 末尾换行
        return content_buffer, []
    elif tool_calls_acc:
        tool_calls = []
        for idx in sorted(tool_calls_acc.keys()):
            entry = tool_calls_acc[idx]
            tool_calls.append({
                "id": entry["id"],
                "type": "function",
                "function": {
                    "name": entry["function"]["name"],
                    "arguments": entry["function"]["arguments"],
                },
            })
        return None, tool_calls

    # 无文本也无工具调用（可能是流式传输异常）
    if stream_error:
        console.print(f"\n  [yellow]⚠️ 流式响应不完整: {stream_error}[/yellow]")
    return None, []


def _process_non_streaming_response(messages: list[dict]) -> tuple[str | None, list[dict], any]:
    """
    使用非流式模式调用 API 并处理响应。

    返回:
        (text_content, tool_calls_list, usage)
    """
    response = call_api_with_retry(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=4096,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    choice = response.choices[0]
    usage = response.usage

    if usage:
        console.print(
            f"  [dim]📊 输入 {usage.prompt_tokens} | "
            f"输出 {usage.completion_tokens} | "
            f"总计 {usage.total_tokens} tokens[/dim]"
        )

    msg = choice.message

    # 工具调用
    if msg.tool_calls and len(msg.tool_calls) > 0:
        tool_calls = [tc.model_dump() for tc in msg.tool_calls]
        return None, tool_calls, usage

    # 文本响应
    content = msg.content or ""
    return content, [], usage


# ============================================================
# 10. 主循环 —— Agent 的核心运行逻辑
# ============================================================

def run_agent():
    """Agent 主循环。"""

    # ----- 启动检测 -----
    print_banner()

    with console.status("[bold yellow]正在检测项目信息...[/bold yellow]", spinner="dots"):
        project_info = detect_project()

    system_prompt = build_system_prompt(project_info)

    # ----- 检查是否有自动保存的会话可以恢复 -----
    messages: list[dict] = [
        {"role": "system", "content": system_prompt}
    ]

    if has_auto_saved_session():
        console.print()
        answer = Prompt.ask(
            "  [bold cyan]检测到上次会话记录，是否恢复？[/bold cyan]",
            choices=["y", "n"],
            default="y",
        )
        if answer.lower() == "y":
            loaded = load_session("__auto__")
            if loaded:
                # 替换 system prompt（system prompt 可能已变化）
                loaded[0] = {"role": "system", "content": system_prompt}
                messages = loaded
                console.print(
                    f"  [green]✅ 已恢复上次会话（{len(messages)} 条消息）[/green]"
                )

    # 会话开始时间（用于 /stats）
    session_start = time.time()

    console.print()
    console.print("[dim]💬 开始对话...[/dim]")
    console.print("[dim]" + "─" * 55 + "[/dim]")

    # ========== 外层循环：接收用户输入 ==========
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]▸[/bold green] 你")
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            auto_save(messages)
            print_goodbye()
            break

        # ----- 特殊命令 -----
        if user_input.lower() == "/exit":
            auto_save(messages)
            print_goodbye()
            break

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            clear_confirm_cache()
            console.print("[dim]🧹 上下文已清空[/dim]")
            continue

        if user_input.lower() == "/help":
            print_help()
            continue

        if user_input.lower() == "/stats":
            print_stats(messages, session_start)
            continue

        if user_input.lower() == "/sessions":
            print_sessions()
            continue

        if user_input.lower().startswith("/save"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("  [yellow]用法: /save <会话名称>[/yellow]")
            else:
                name = parts[1].strip()
                if save_session(name, messages):
                    console.print(f"  [green]✅ 会话已保存为 '{name}'[/green]")
                else:
                    console.print(f"  [red]❌ 保存会话 '{name}' 失败[/red]")
            continue

        if user_input.lower().startswith("/load"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                console.print("  [yellow]用法: /load <会话名称>[/yellow]")
            else:
                name = parts[1].strip()
                loaded = load_session(name)
                if loaded is not None:
                    messages.clear()
                    messages.extend(loaded)
                    console.print(f"  [green]✅ 已加载会话 '{name}'（{len(messages)} 条消息）[/green]")
                else:
                    console.print(f"  [red]❌ 会话 '{name}' 不存在[/red]")
            continue

        if not user_input:
            continue

        # ----- 每轮对话开始，清空命令确认缓存 -----
        clear_confirm_cache()

        # ----- 把用户消息加入对话历史 -----
        messages.append({"role": "user", "content": user_input})

        # ----- 智能上下文压缩 -----
        messages = compress_messages(messages)

        # ----- 自动保存当前会话 -----
        auto_save(messages)

        # ========== 内层循环：工具调用往返 ==========
        tool_rounds = 0

        while tool_rounds < MAX_TOOL_ROUNDS:
            try:
                if STREAMING_ENABLED:
                    # 流式模式：不用 console.status()，避免 spinner 阻塞输出
                    console.print(
                        f"  [dim]🤔 思考中... (轮次 {tool_rounds + 1})[/dim]"
                    )
                    console.print()
                    text_content, tool_calls = _process_streaming_response(messages)
                    finish_reason = None
                else:
                    with console.status(
                        f"[bold yellow]正在思考... (轮次 {tool_rounds + 1})[/bold yellow]",
                        spinner="dots",
                    ):
                        text_content, tool_calls, usage = _process_non_streaming_response(messages)

            except AgentAPIError as e:
                console.print(f"  [bold red]❌ API 调用失败: {e}[/bold red]")
                # 如果已经重试过还不行，尝试无工具的无流式调用作为 fallback
                console.print("  [yellow]⚠️ 尝试降级调用（无工具）...[/yellow]")
                try:
                    response = call_api_with_retry(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=4096,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                    )
                    fallback = response.choices[0].message.content or "（无响应）"
                    messages.append({"role": "assistant", "content": fallback})
                    console.print()
                    console.print("[dim]" + "─" * 55 + "[/dim]")
                    console.print(Panel(fallback, title="[bold white]🤖 Agent[/bold white]", border_style="cyan"))
                except Exception as fallback_err:
                    messages.append({
                        "role": "assistant",
                        "content": f"抱歉，API 调用出错了: {e}"
                    })
                break
            except Exception as e:
                console.print(f"  [bold red]❌ 未预期的 API 错误: {e}[/bold red]")
                messages.append({
                    "role": "assistant",
                    "content": f"抱歉，API 调用出错了: {e}"
                })
                break

            # --- 情况 1：模型想调用工具 ---
            if tool_calls:
                # 构造 assistant 消息
                assistant_msg = {
                    "role": "assistant",
                    "content": text_content,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_msg)

                for tc in tool_calls:
                    result = execute_tool_call(tc)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                tool_rounds += 1
                continue

            # --- 情况 2：模型正常返回文本 ---
            if text_content:
                messages.append({
                    "role": "assistant",
                    "content": text_content,
                })

                # 分隔线 + 回复（流式模式下文本已经打印过了）
                if not STREAMING_ENABLED:
                    console.print()
                    console.print("[dim]" + "─" * 55 + "[/dim]")
                    console.print(
                        Panel(
                            text_content,
                            title="[bold white]🤖 Agent[/bold white]",
                            border_style="cyan",
                        )
                    )
                else:
                    console.print()
                    console.print("[dim]" + "─" * 55 + "[/dim]")
                break

            # --- 情况 3：既无文本也无工具调用 ---
            console.print("  [yellow]⚠️ 模型未返回有效响应[/yellow]")
            messages.append({
                "role": "assistant",
                "content": "[注意] 模型未返回有效响应，请重试。"
            })
            break

        # ----- 达到最大工具轮数 -----
        if tool_rounds >= MAX_TOOL_ROUNDS:
            console.print(
                f"\n  [yellow]⚠️ 已达到最大工具调用轮数 ({MAX_TOOL_ROUNDS})，强制结束本轮[/yellow]"
            )
            messages.append({
                "role": "user",
                "content": "已达到工具调用上限，请基于已有的信息给出回答。"
            })
            try:
                if STREAMING_ENABLED:
                    console.print("  [dim]🤔 正在生成总结...[/dim]")
                    console.print()
                    text_content, _ = _process_streaming_response(messages)
                    final_reply = text_content or "（无响应）"
                else:
                    with console.status("[bold yellow]正在生成总结...[/bold yellow]", spinner="dots"):
                        response = call_api_with_retry(
                            model=MODEL_NAME,
                            messages=messages,
                            max_tokens=4096,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                        )
                        final_reply = response.choices[0].message.content or ""

                messages.append({"role": "assistant", "content": final_reply})
                console.print()
                console.print("[dim]" + "─" * 55 + "[/dim]")
                console.print(
                    Panel(
                        final_reply,
                        title="[bold white]🤖 Agent[/bold white]",
                        border_style="cyan",
                    )
                )
            except Exception as e:
                console.print(f"  [bold red]❌ 最终总结调用失败: {e}[/bold red]")


# ============================================================
# 11. 程序入口
# ============================================================

if __name__ == "__main__":
    run_agent()
