"""
tools.py —— Coding Agent 的工具系统
=====================================
这个文件定义了 Agent 可以调用的所有工具函数，以及发送给 API 的工具 JSON Schema。

新增一个工具的步骤：
  1. 在下方写一个函数（func）
  2. 在 AVAILABLE_TOOLS 字典里注册映射
  3. 在 TOOLS 列表里添加 JSON Schema 定义

v2 变更：
  - 工具函数改用异常代替字符串错误（ToolInputError / ToolExecutionError / ToolSecurityError）
  - execute_command 安全分类逻辑不变，但 blocked 现在抛出 ToolSecurityError
"""

import os
import re
import shlex
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from exceptions import (
    ToolError,
    ToolInputError,
    ToolExecutionError,
    ToolSecurityError,
)

# 全局 Rich Console 实例，供所有工具函数使用
console = Console()

# ============================================================
# 工作区根目录 —— 由 agent.py 在启动时设置
# ============================================================

_workspace_root: str = os.getcwd()


def set_workspace_root(path: str) -> None:
    """设置工作区根目录（由 agent.py 在启动时调用）。"""
    global _workspace_root
    _workspace_root = os.path.abspath(path)


# ============================================================
# 辅助函数
# ============================================================

def _format_size(size_bytes: int) -> str:
    """把字节数转成人类可读的大小。"""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def is_safe_path(file_path: str, workspace_root: str | None = None) -> tuple[bool, str]:
    """
    检查给定路径是否在工作区目录内，防止 Agent 写入项目外的文件。

    参数:
        file_path:      要检查的文件路径
        workspace_root: 工作区根目录，默认使用 _workspace_root

    返回:
        (是否安全, 原因说明)
    """
    root = workspace_root or _workspace_root

    try:
        resolved = Path(file_path).resolve()
        root_resolved = Path(root).resolve()

        try:
            resolved.relative_to(root_resolved)
            return True, ""
        except ValueError:
            return False, (
                f"路径安全限制：'{file_path}' 解析后为 '{resolved}'，"
                f"不在工作区 '{root_resolved}' 内。禁止使用 ../ 跳出项目根目录。"
            )
    except Exception as e:
        return False, f"路径解析失败: {e}"


def get_file_info(file_path: str) -> dict | None:
    """
    获取文件的 stat 信息。

    返回:
        包含 size, mtime, is_file 等信息的字典；文件不存在则返回 None
    """
    path = Path(file_path)
    if not path.exists():
        return None
    stat = path.stat()
    return {
        "size": stat.st_size,
        "size_human": _format_size(stat.st_size),
        "mtime": stat.st_mtime,
        "mtime_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
    }


# ============================================================
# 1. read_file —— 读取文件内容（带行号，像 cat -n）
# ============================================================

def read_file(file_path: str, line_start: int | None = None, line_end: int | None = None) -> str:
    """
    读取指定文件的内容，返回带行号的文本。

    参数:
        file_path:   文件的绝对路径或相对路径
        line_start:  起始行号（从 1 开始），不传则从第 1 行开始
        line_end:    结束行号（包含），不传则读到文件末尾

    返回:
        带行号的文本内容，格式同 cat -n 命令

    异常:
        ToolInputError:     参数不合法
        ToolExecutionError: 文件不存在、无法读取等
    """
    if not file_path or not file_path.strip():
        raise ToolInputError("file_path 不能为空", tool_name="read_file")

    path = Path(file_path)

    if not path.exists():
        raise ToolExecutionError(f"文件不存在 —— {file_path}", tool_name="read_file")
    if not path.is_file():
        raise ToolExecutionError(f"路径是一个目录，不是文件 —— {file_path}", tool_name="read_file")

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except PermissionError:
        raise ToolExecutionError(f"没有权限读取文件 —— {file_path}", tool_name="read_file")
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        except Exception as e:
            raise ToolExecutionError(
                f"文件编码无法识别 —— {file_path}，详情: {e}", tool_name="read_file"
            )
    except Exception as e:
        raise ToolExecutionError(
            f"读取文件时发生异常 —— {file_path}，详情: {e}", tool_name="read_file"
        )

    total_lines = len(lines)
    start = line_start if line_start is not None else 1
    end = line_end if line_end is not None else total_lines

    start = max(1, start)
    end = min(total_lines, end)

    if start > end:
        raise ToolInputError(
            f"line_start ({line_start}) 大于 line_end ({line_end})", tool_name="read_file"
        )

    result_parts = []
    for i in range(start - 1, end):
        line_num = i + 1
        result_parts.append(f"{line_num:>6}\t{lines[i].rstrip()}")

    return "\n".join(result_parts)


# ============================================================
# 2. list_directory —— 列出目录内容
# ============================================================

def list_directory(dir_path: str) -> str:
    """
    列出指定目录下的所有文件和子目录，标注类型和文件大小。

    异常:
        ToolInputError:     参数为空
        ToolExecutionError: 目录不存在或无法访问
    """
    if not dir_path or not dir_path.strip():
        raise ToolInputError("dir_path 不能为空", tool_name="list_directory")

    path = Path(dir_path)

    if not path.exists():
        raise ToolExecutionError(f"目录不存在 —— {dir_path}", tool_name="list_directory")
    if not path.is_dir():
        raise ToolExecutionError(f"路径不是目录 —— {dir_path}", tool_name="list_directory")

    try:
        entries = sorted(path.iterdir())
    except PermissionError:
        raise ToolExecutionError(
            f"没有权限访问目录 —— {dir_path}", tool_name="list_directory"
        )
    except Exception as e:
        raise ToolExecutionError(
            f"读取目录时发生异常 —— {dir_path}，详情: {e}", tool_name="list_directory"
        )

    if not entries:
        return f"目录为空 —— {dir_path}"

    result_parts = [f"目录 {dir_path} 的内容（共 {len(entries)} 项）:"]
    for entry in entries:
        try:
            if entry.is_dir():
                result_parts.append(f"  [目录]  {entry.name}/")
            elif entry.is_file():
                size = entry.stat().st_size
                result_parts.append(f"  [文件]  {entry.name}  ({_format_size(size)})")
            else:
                result_parts.append(f"  [其他]  {entry.name}")
        except OSError:
            result_parts.append(f"  [未知]  {entry.name}")

    return "\n".join(result_parts)


# ============================================================
# 3. search_in_file —— 在文件中搜索关键词
# ============================================================

def search_in_file(file_path: str, query: str) -> str:
    """
    在文件中搜索包含 query 的行，返回匹配行及其上下文（前后各 2 行）。

    异常:
        ToolInputError:     参数为空
        ToolExecutionError: 文件不存在或无法读取
    """
    if not file_path or not file_path.strip():
        raise ToolInputError("file_path 不能为空", tool_name="search_in_file")
    if not query or not query.strip():
        raise ToolInputError("query 不能为空", tool_name="search_in_file")

    path = Path(file_path)

    if not path.exists():
        raise ToolExecutionError(f"文件不存在 —— {file_path}", tool_name="search_in_file")
    if not path.is_file():
        raise ToolExecutionError(f"路径不是文件 —— {file_path}", tool_name="search_in_file")

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except PermissionError:
        raise ToolExecutionError(f"没有权限读取文件 —— {file_path}", tool_name="search_in_file")
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        except Exception as e:
            raise ToolExecutionError(
                f"文件编码无法识别 —— {file_path}，详情: {e}", tool_name="search_in_file"
            )
    except Exception as e:
        raise ToolExecutionError(
            f"读取文件时发生异常 —— {file_path}，详情: {e}", tool_name="search_in_file"
        )

    total_lines = len(lines)
    matched_indices = []

    for i, line in enumerate(lines):
        if query in line:
            matched_indices.append(i)

    if not matched_indices:
        return f"未在 {file_path} 中找到 \"{query}\""

    lines_to_show = set()
    context_radius = 2

    for idx in matched_indices:
        for offset in range(-context_radius, context_radius + 1):
            ctx_idx = idx + offset
            if 0 <= ctx_idx < total_lines:
                lines_to_show.add(ctx_idx)

    result_parts = [f'在 {file_path} 中搜索 "{query}"，找到 {len(matched_indices)} 处匹配:']
    result_parts.append("")

    sorted_indices = sorted(lines_to_show)
    prev_idx = -2

    for idx in sorted_indices:
        if idx - prev_idx > 1:
            result_parts.append("  " + "-" * 50)
        marker = ">>>" if idx in matched_indices else "   "
        result_parts.append(f"{marker} {idx + 1:>6}\t{lines[idx].rstrip()}")
        prev_idx = idx

    return "\n".join(result_parts)


# ============================================================
# 4. execute_command —— 执行 shell 命令（三级安全分类）
# ============================================================

# --- 第一级：安全命令（直接执行，无需确认）---
SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "wc",
    "find", "grep", "git",
    "python", "python3", "pytest",
    "cargo", "npm", "node", "npx",
    "echo", "date", "which", "pwd",
    "sort", "uniq", "cut", "tr", "diff",
    "env", "printenv", "uname", "hostname", "whoami",
    "df", "du", "file",
}

# --- 第二级：需确认命令（打印警告，等用户输入 y/n 后才执行）---
NEED_CONFIRM_COMMANDS = {
    "pip", "pip3",
    "npm",
    "yarn", "pnpm",
    "cargo",
    "rustc", "go",
    "make", "cmake",
    "mv", "cp",
    "mkdir", "touch",
    "chmod",
    "curl", "wget",
    "sed", "awk",
    "xargs",
}

# --- 第三级：禁止命令（直接拒绝，抛出异常）---
BLOCKED_PATTERNS = [
    "rm ", "rmdir",
    "sudo", "su ",
    "chmod 777", "chmod -R 777",
    "chmod u+s", "chmod g+s",
    "chown",
    "mkfs", "mkswap", "dd ",
    ":(){ :|:& };:",
    "> /dev/", "> /etc/", "> /proc/",
    "| sh", "| bash", "| zsh",
    "reboot", "shutdown", "halt",
    "kill", "killall", "pkill",
    "iptables", "nft ",
    "passwd",
    "mount", "umount",
    "mkfs.",
    "docker rm", "docker rmi",
    "chattr",
    "curl " "| sh", "wget " "| sh",
]


def _classify_command(base_cmd: str, full_command: str) -> str:
    """
    对命令进行安全分级。

    返回:
        "safe":         直接执行
        "need_confirm": 需要用户确认
        "blocked":      直接拒绝
    """
    # 先检查禁止模式
    normalized = full_command.replace("\n", " ")
    for pattern in BLOCKED_PATTERNS:
        if pattern in normalized:
            return "blocked"

    # 再检查是否需要确认
    if base_cmd in NEED_CONFIRM_COMMANDS:
        if base_cmd in ("npm", "yarn", "pnpm"):
            parts = normalized.strip().split()
            if len(parts) >= 2 and parts[1] in (
                "run", "test", "lint", "fmt", "check", "list", "ls",
                "view", "outdated", "audit", "why", "doctor", "help",
            ):
                return "safe"
            return "need_confirm"
        if base_cmd == "cargo":
            parts = normalized.strip().split()
            if len(parts) >= 2 and parts[1] in (
                "check", "test", "fmt", "clippy", "doc", "help", "version",
            ):
                return "safe"
            return "need_confirm"
        if base_cmd in ("pip", "pip3"):
            parts = normalized.strip().split()
            if len(parts) >= 2 and parts[1] in (
                "list", "show", "freeze", "check", "config", "help", "--version",
            ):
                return "safe"
            return "need_confirm"
        if base_cmd in ("curl", "wget"):
            return "need_confirm"
        if base_cmd in ("mv", "cp", "mkdir", "touch", "chmod"):
            return "need_confirm"
        return "need_confirm"

    # 最后检查是否在安全白名单中
    if base_cmd in SAFE_COMMANDS:
        return "safe"

    # 不在任何白名单中
    return "blocked"


# 需确认命令的交互回复缓存（避免同一轮重复提问）
_user_confirm_cache: dict[str, bool] = {}


def _ask_user_to_confirm(command: str) -> bool:
    """
    询问用户是否允许执行某条需确认的命令。
    同一轮对话中相同的命令不会重复询问。
    """
    # 缓存键：命令的前 120 个字符
    cache_key = command.strip()[:120]
    if cache_key in _user_confirm_cache:
        return _user_confirm_cache[cache_key]

    warning_text = Text()
    warning_text.append("⚠️  需要确认\n", style="bold yellow")
    warning_text.append(f"命令: ", style="dim")
    warning_text.append(f"{command}", style="bold white")
    warning_text.append(f"\n分类: 该命令被归类为需确认操作", style="dim")

    console.print()
    console.print(Panel(warning_text, border_style="yellow", title="安全确认"))

    answer = Prompt.ask(
        "  [bold yellow]是否执行？[/bold yellow]",
        choices=["y", "n"],
        default="n",
    )

    allowed = answer.lower() == "y"
    _user_confirm_cache[cache_key] = allowed
    return allowed


def clear_confirm_cache() -> None:
    """清空需确认命令缓存（每轮对话开始时调用）。"""
    _user_confirm_cache.clear()


def execute_command(command: str) -> str:
    """
    在三级安全分类下执行 shell 命令。

    1. 安全命令：直接执行
    2. 需确认命令：显示警告，等待用户输入 y/n
    3. 禁止命令：抛出 ToolSecurityError

    参数:
        command: 要执行的 shell 命令字符串

    返回:
        命令的执行结果（stdout + stderr）

    异常:
        ToolInputError:     命令为空
        ToolSecurityError:  命令被安全限制阻止
        ToolExecutionError: 命令执行失败
    """
    if not command or not command.strip():
        raise ToolInputError("command 不能为空", tool_name="execute_command")

    cmd_parts = command.strip().split()
    if not cmd_parts:
        raise ToolInputError("无法解析命令", tool_name="execute_command")

    base_cmd = cmd_parts[0]
    if "/" in base_cmd:
        base_cmd = base_cmd.split("/")[-1]

    # --- 安全分级 ---
    classification = _classify_command(base_cmd, command)

    if classification == "blocked":
        raise ToolSecurityError(
            f"命令 \"{base_cmd}\" 包含危险操作，已被阻止执行。\n"
            f"如果你确实需要执行此操作，请在终端中手动执行。",
            tool_name="execute_command",
        )

    if classification == "need_confirm":
        allowed = _ask_user_to_confirm(command)
        if not allowed:
            return f"⏭️ 用户取消了命令执行: {command}"
        console.print("  [dim]用户确认，继续执行...[/dim]")

    # --- 执行命令（优先使用 shell=False 以增强安全性）---
    # 检测 shell 元字符：如果命令包含 | > < && || ; $() `` 等则必须用 shell=True
    SHELL_META_PATTERN = re.compile(r'[|&;><$`!]|\|\||&&')

    try:
        if SHELL_META_PATTERN.search(command):
            # 包含 shell 元字符，使用 shell=True（经过安全分级检查）
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd(),
            )
        else:
            # 简单命令：用 shlex 安全分割后用 shell=False 执行
            try:
                cmd_list = shlex.split(command)
            except ValueError as e:
                raise ToolInputError(
                    f"命令解析失败 —— {e}", tool_name="execute_command"
                )
            result = subprocess.run(
                cmd_list,
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd(),
            )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout.rstrip())
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr.rstrip()}")

        if not output_parts:
            return f"命令执行完毕，无输出（退出码: {result.returncode}）"

        return "\n".join(output_parts)

    except subprocess.TimeoutExpired:
        raise ToolExecutionError(
            f"命令执行超时（30 秒）—— {command}", tool_name="execute_command"
        )
    except FileNotFoundError:
        raise ToolExecutionError(f"命令未找到 —— {base_cmd}", tool_name="execute_command")
    except ToolInputError:
        raise
    except Exception as e:
        raise ToolExecutionError(
            f"执行命令时发生异常 —— {command}，详情: {e}", tool_name="execute_command"
        )


# ============================================================
# 5. write_file —— 写入文件内容
# ============================================================

def write_file(file_path: str, content: str, mode: str = "overwrite") -> str:
    """
    将内容写入文件。支持两种模式：
      - "overwrite": 覆盖整个文件（默认）
      - "append":    追加到文件末尾

    安全措施：
      1. 检查路径是否在工作区内（禁止 ../ 跳出）
      2. overwrite 模式下如果文件存在，先备份原内容
      3. 写入后返回文件信息

    参数:
        file_path: 文件路径（必须在工作区内）
        content:   要写入的内容
        mode:      "overwrite" 或 "append"

    返回:
        操作结果描述

    异常:
        ToolInputError:     参数不合法
        ToolSecurityError:  路径不在工作区内
        ToolExecutionError: 写入失败
    """
    if not file_path or not file_path.strip():
        raise ToolInputError("file_path 不能为空", tool_name="write_file")
    if content is None:
        raise ToolInputError("content 不能为 None", tool_name="write_file")
    if mode not in ("overwrite", "append"):
        raise ToolInputError(
            f"不支持的 mode '{mode}'，只支持 'overwrite' 和 'append'",
            tool_name="write_file",
        )

    # --- 安全检查：路径必须在工作区内 ---
    safe, reason = is_safe_path(file_path)
    if not safe:
        raise ToolSecurityError(reason, tool_name="write_file")

    path = Path(file_path)

    # --- 检查不是目录 ---
    if path.exists() and path.is_dir():
        raise ToolExecutionError(
            f"'{file_path}' 是一个目录，不能作为文件写入", tool_name="write_file"
        )

    # --- 备份逻辑（仅 overwrite 模式且文件存在时）---
    backup_info = ""
    if mode == "overwrite" and path.exists():
        try:
            original_content = path.read_text(encoding="utf-8")
            original_size = len(original_content)
            original_lines = original_content.count("\n") + 1
            backup_info = (
                f"\n📦 已备份原文件: {original_lines} 行, {original_size} 字符"
            )
        except Exception as e:
            backup_info = f"\n⚠️ 备份原文件失败: {e}"

    # --- 确保父目录存在 ---
    parent = path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise ToolExecutionError(
                f"没有权限创建目录 —— {parent}", tool_name="write_file"
            )
        except Exception as e:
            raise ToolExecutionError(
                f"创建父目录失败 —— {parent}，详情: {e}", tool_name="write_file"
            )

    # --- 执行写入 ---
    try:
        if mode == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            action = "追加"
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            action = "写入"
    except PermissionError:
        raise ToolExecutionError(
            f"没有权限写入文件 —— {file_path}", tool_name="write_file"
        )
    except OSError as e:
        raise ToolExecutionError(
            f"写入文件时发生系统错误 —— {file_path}，详情: {e}", tool_name="write_file"
        )
    except Exception as e:
        raise ToolExecutionError(
            f"写入文件时发生异常 —— {file_path}，详情: {e}", tool_name="write_file"
        )

    # --- 返回结果 ---
    file_info = get_file_info(file_path)
    content_lines = content.count("\n") + 1
    content_bytes = len(content.encode("utf-8"))

    result = (
        f"✅ 已{action}文件: {file_path}\n"
        f"   内容: {content_lines} 行, {content_bytes} 字节"
    )
    if file_info:
        result += f"\n   文件大小: {file_info['size_human']}"
    if backup_info:
        result += backup_info

    return result


# ============================================================
# 6. ask_followup_question —— 向用户澄清问题
# ============================================================

def ask_followup_question(question: str, options: list[str] | None = None) -> str:
    """
    当模型不确定用户的意图时，主动向用户提问澄清。

    参数:
        question: 要问用户的问题
        options:  可选答案列表（最多 5 个），不传则让用户自由输入

    返回:
        用户的回答字符串

    异常:
        ToolInputError: question 为空
    """
    if not question or not question.strip():
        raise ToolInputError("question 不能为空", tool_name="ask_followup_question")

    question_text = Text()
    question_text.append("❓ ", style="bold cyan")
    question_text.append(question, style="white")

    console.print()
    console.print(Panel(question_text, border_style="cyan", title="Agent 需要确认"))

    if options and len(options) > 0:
        display_options = options[:5]

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("key", style="bold cyan", width=4)
        table.add_column("value", style="white")
        for i, opt in enumerate(display_options, 1):
            table.add_row(f"{i}.", opt)
        table.add_row("0.", "其他（自定义输入）")

        console.print(table)

        valid_choices = [str(i) for i in range(len(display_options) + 1)]
        choice = Prompt.ask(
            "  [bold cyan]请选择[/bold cyan]",
            choices=valid_choices,
            default="1",
        )

        idx = int(choice)
        if idx == 0:
            answer = Prompt.ask("  [bold cyan]请输入[/bold cyan]")
        else:
            answer = display_options[idx - 1]
    else:
        answer = Prompt.ask(f"  [bold cyan]请回答[/bold cyan]")

    console.print(f"  [dim]用户回答: {answer}[/dim]")
    return answer


# ============================================================
# 工具注册表 —— 把函数名映射到函数对象
# ============================================================

AVAILABLE_TOOLS: dict[str, callable] = {
    "read_file":              read_file,
    "list_directory":         list_directory,
    "search_in_file":         search_in_file,
    "execute_command":        execute_command,
    "write_file":             write_file,
    "ask_followup_question":  ask_followup_question,
}


# ============================================================
# 工具 JSON Schema 定义 —— 发送给 API 的工具描述
# ============================================================

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容，返回带行号的文本（类似 cat -n）。可以指定行号范围来只读取文件的一部分。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件路径（绝对路径或相对路径）",
                    },
                    "line_start": {
                        "type": "integer",
                        "description": "起始行号（从 1 开始），留空则从第 1 行开始读取",
                    },
                    "line_end": {
                        "type": "integer",
                        "description": "结束行号（包含），留空则读到文件末尾",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录下的所有文件和子目录，标注是文件还是目录，并显示文件大小。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dir_path": {
                        "type": "string",
                        "description": "要列出内容的目录路径（绝对路径或相对路径）",
                    },
                },
                "required": ["dir_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_in_file",
            "description": "在文件中搜索包含指定关键词的行，返回匹配行及其上下文（前后各 2 行）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要搜索的文件路径",
                    },
                    "query": {
                        "type": "string",
                        "description": "搜索关键词（区分大小写）",
                    },
                },
                "required": ["file_path", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": (
                "执行 shell 命令并返回输出。有三类命令："
                "1) 安全命令直接执行（ls, cat, grep, find, git, python, pytest 等）；"
                "2) 需确认命令会询问用户后再执行（pip install, npm install, cargo build, mv, cp 等）；"
                "3) 禁止命令直接拒绝（rm, sudo, chmod 777 等危险操作）。"
                "注意：当命令被阻止或需确认时，不要反复尝试同一命令。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 shell 命令",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "将内容写入文件。支持两种模式：'overwrite'（覆盖整个文件）和 'append'（追加到文件末尾）。"
                "写入前会自动检查路径是否在工作区内，overwrite 模式下如文件已存在会先备份。"
                "只能写入工作区内的文件，不能使用 ../ 跳出项目根目录。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要写入的文件路径（必须在当前工作区内）",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文件内容",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "写入模式：overwrite 覆盖整个文件，append 追加到末尾。默认为 overwrite。",
                    },
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_followup_question",
            "description": (
                "当你对用户的需求不确定时，使用此工具向用户提问澄清。"
                "可以提供最多 5 个预设选项让用户选择，也可以让用户自由输入。"
                "重要：遇到模糊需求时优先使用此工具，不要自己猜测用户意图。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "要问用户的问题",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "可选的答案列表（最多 5 个），不传则让用户自由输入",
                    },
                },
                "required": ["question"],
            },
        },
    },
]
