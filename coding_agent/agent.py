"""
agent.py —— Coding Agent 主入口
================================
这个文件实现了 Agent 的核心循环逻辑：
  1. 加载配置（.env）
  2. 初始化 OpenAI 兼容客户端
  3. 项目自动感知（git 状态、语言检测、结构摘要）
  4. 维护对话历史 + 智能上下文压缩（三级策略）
  5. 调用 LLM API，处理工具调用

运行方式：
  python agent.py

依赖安装：
  pip install openai python-dotenv tiktoken rich
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# 导入我们的工具系统
from tools import (
    AVAILABLE_TOOLS,
    TOOLS,
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
# 上下文 token 上限（默认 80000，大多数模型的保守值）
CONTEXT_LIMIT_TOKENS = int(os.getenv("CONTEXT_LIMIT_TOKENS", "80000"))

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

    console.print()
    console.print(Panel(
        info_table,
        title="[bold white]🤖 Coding Agent[/bold white]",
        subtitle="[dim]输入 /exit 退出 | /clear 清空上下文[/dim]",
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
            # 获取当前分支名
            branch = sp.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, timeout=5, cwd=WORKSPACE_ROOT
            )
            branch_name = branch.stdout.strip() or "unknown"

            # 获取最近 3 条 commit 摘要
            log = sp.run(
                ["git", "log", "--oneline", "-3", "--format=%s"],
                capture_output=True, text=True, timeout=5, cwd=WORKSPACE_ROOT
            )
            commits = [c.strip() for c in log.stdout.strip().split("\n") if c.strip()]

            parts.append(f"Git 分支: {branch_name}")
            if commits:
                parts.append(f"最近提交: {' | '.join(commits[:3])}")
    except Exception:
        pass  # git 检测失败不阻塞启动

    # --- 3.2 项目语言检测 ---
    try:
        ext_counter = Counter()
        # 标志文件检测（优先级高于扩展名计数）
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

            # 统计扩展名
            if entry.is_file():
                ext = entry.suffix.lower()
                if ext:
                    ext_counter[ext] += 1

        # 也扫描一级子目录
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

        # 推断语言
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

        # 优先使用标志文件检测结果
        if detected_flags:
            parts.append(f"项目类型: {', '.join(detected_flags)}")
        elif ext_counter:
            # 找出现次数最多的扩展名
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
        for entry in root_entries[:30]:  # 最多显示 30 项
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
    用 tiktoken 估算消息列表的总 token 数。

    使用 cl100k_base 编码器（GPT-4 / DeepSeek 通用），
    估算结果不需要精确到个位，用于判断是否需要压缩。
    """
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
        # fallback: 字符数 / 2
        total_chars = sum(len(str(m)) for m in messages)
        return total_chars // 2


# ============================================================
# 6. 智能上下文压缩 —— 三级策略
# ============================================================

def _summarize_with_llm(messages_to_summarize: list[dict]) -> str:
    """
    调用模型本身对历史消息做摘要。
    使用 temperature=0 追求稳定输出。
    """
    summary_prompt = (
        "请用中文总结以下对话历史，包含四个部分：\n"
        "1. 用户曾要求做什么\n"
        "2. 已完成的操作\n"
        "3. 涉及的关键文件\n"
        "4. 未解决的问题\n\n"
        "保持简洁，每部分用一句话概括。"
    )

    # 构造摘要请求的消息列表
    summary_messages = messages_to_summarize + [
        {"role": "user", "content": summary_prompt}
    ]

    try:
        response = client.chat.completions.create(
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
            return messages  # 轮数不够，不压缩

        cut_idx = user_indices[-(keep_rounds)]
        recent_msgs = other_msgs[cut_idx:]
        older_msgs = other_msgs[:cut_idx]

        # 对更早的消息：保留 user 原文，截断过长的 assistant 回复
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
                # tool 结果只保留前 200 字符
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > 200:
                    new_msg = msg.copy()
                    new_msg["content"] = content[:200] + "\n...(已截断)"
                    truncated.append(new_msg)
                else:
                    truncated.append(msg)

        # 插入摘要提示
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
        # 对话轮数不够，但 token 已经很高了，强制只保留最近 2 轮
        pass

    cut_idx = user_indices[-(keep_rounds)] if len(user_indices) >= keep_rounds else 0
    recent_msgs = other_msgs[cut_idx:]
    older_msgs = other_msgs[:cut_idx]

    # 调用模型做摘要
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
# 7. 执行单个工具调用（带 rich 日志）
# ============================================================

def execute_tool_call(tool_call: dict) -> str:
    """
    根据 API 返回的 tool_call 字典，找到对应的函数并执行。
    使用 rich 输出彩色日志。
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

    # --- 调用工具函数 ---
    func = AVAILABLE_TOOLS[func_name]

    try:
        result = func(**arguments)
    except TypeError as e:
        return f"错误：参数不匹配 —— {e}，收到参数: {arguments}"
    except Exception as e:
        return f"错误：工具执行异常 —— {e}"

    # --- 截断过长的结果 ---
    max_result_len = 8000
    if len(result) > max_result_len:
        result = result[:max_result_len] + (
            f"\n\n...(结果过长，已截断，原长度 {len(result)} 字符)"
        )

    # --- 结果日志：根据成功/失败用不同颜色 ---
    if result.startswith("错误") or result.startswith("⛔"):
        color = "red"
    elif result.startswith("⚠") or result.startswith("⏭"):
        color = "yellow"
    else:
        color = "green"

    console.print(f"  [{color}]📋 返回 {len(result)} 字符[/{color}]")

    return result


# ============================================================
# 8. 主循环 —— Agent 的核心运行逻辑
# ============================================================

def run_agent():
    """Agent 主循环。"""

    # ----- 启动检测 -----
    print_banner()

    with console.status("[bold yellow]正在检测项目信息...[/bold yellow]", spinner="dots"):
        project_info = detect_project()

    system_prompt = build_system_prompt(project_info)

    # 初始化消息列表
    messages: list[dict] = [
        {"role": "system", "content": system_prompt}
    ]

    console.print()
    console.print("[dim]💬 开始对话...[/dim]")
    console.print("[dim]" + "─" * 55 + "[/dim]")

    # ========== 外层循环：接收用户输入 ==========
    while True:
        try:
            # 绿色提示符
            user_input = Prompt.ask("\n[bold green]▸[/bold green] 你")
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print_goodbye()
            break

        # ----- 特殊命令 -----
        if user_input.lower() == "/exit":
            print_goodbye()
            break

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system_prompt}]
            console.print("[dim]🧹 上下文已清空[/dim]")
            continue

        if not user_input:
            continue

        # ----- 把用户消息加入对话历史 -----
        messages.append({"role": "user", "content": user_input})

        # ----- 智能上下文压缩（每次添加用户消息后检查）-----
        messages = compress_messages(messages)

        # ========== 内层循环：工具调用往返 ==========
        tool_rounds = 0

        while tool_rounds < MAX_TOOL_ROUNDS:
            # --- 调用 LLM API（带 spinner 动画）---
            try:
                with console.status(
                    f"[bold yellow]正在思考... (轮次 {tool_rounds + 1})[/bold yellow]",
                    spinner="dots",
                ):
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        max_tokens=4096,
                    )
            except Exception as e:
                console.print(f"  [bold red]❌ API 调用失败: {e}[/bold red]")
                messages.append({
                    "role": "assistant",
                    "content": f"抱歉，API 调用出错了: {e}"
                })
                break

            # --- 解析 API 响应 ---
            choice = response.choices[0]
            finish_reason = choice.finish_reason

            # Token 用量
            usage = response.usage
            if usage:
                console.print(
                    f"  [dim]📊 输入 {usage.prompt_tokens} | "
                    f"输出 {usage.completion_tokens} | "
                    f"总计 {usage.total_tokens} tokens[/dim]"
                )

            # --- 情况 1：模型想调用工具 ---
            if finish_reason == "tool_calls" or (
                choice.message.tool_calls and len(choice.message.tool_calls) > 0
            ):
                assistant_msg = choice.message.model_dump(exclude_none=True)
                messages.append(assistant_msg)

                for tool_call in choice.message.tool_calls:
                    result = execute_tool_call(tool_call.model_dump())
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                tool_rounds += 1
                continue

            # --- 情况 2：模型正常返回文本 ---
            if finish_reason == "stop" or choice.message.content:
                assistant_content = choice.message.content or ""
                messages.append({
                    "role": "assistant",
                    "content": assistant_content,
                })

                # 分隔线 + 回复
                console.print()
                console.print("[dim]" + "─" * 55 + "[/dim]")
                console.print(
                    Panel(
                        assistant_content,
                        title="[bold white]🤖 Agent[/bold white]",
                        border_style="cyan",
                    )
                )
                break

            # --- 情况 3：其他异常 ---
            console.print(f"  [yellow]⚠️ 异常结束原因: {finish_reason}[/yellow]")
            messages.append({
                "role": "assistant",
                "content": f"[注意] API 返回异常的 finish_reason: {finish_reason}"
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
                with console.status("[bold yellow]正在生成总结...[/bold yellow]", spinner="dots"):
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        max_tokens=4096,
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
# 9. 程序入口
# ============================================================

if __name__ == "__main__":
    run_agent()
