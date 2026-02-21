#!/usr/bin/env python3
"""Local GLM-5 coding and research agent for Modal OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://api.us-west-2.modal.direct/v1"
DEFAULT_MODEL = "zai-org/GLM-5-FP8"
MAX_TOOL_RESULT_CHARS = 16000
DEFAULT_SHELL_TIMEOUT = 120
DEFAULT_EXEC_TIMEOUT = 120

SYSTEM_PROMPT = """You are a high-agency local coding and research agent.

Rules:
- Always prefer direct, actionable execution through tools.
- For coding tasks: inspect files first, then edit precisely, then verify via commands.
- For research tasks: search, fetch, compare sources, and cite URLs in final answer.
- Keep answers concise and structured.
- Do not use emojis unless the user explicitly asks for them.
- For publish/merge actions, prefer preview mode first and execute only when user explicitly requests it.
- Never fabricate tool output.
- Respect workspace boundaries and avoid destructive commands unless explicitly requested.
"""

DANGEROUS_PATTERNS = [
    r"(^|\s)sudo(\s|$)",
    r"(^|\s)shutdown(\s|$)",
    r"(^|\s)reboot(\s|$)",
    r"rm\s+-rf\s+/($|\s)",
    r"(^|\s)mkfs(\.|\s|$)",
    r"(^|\s)dd\s+if=",
]


def load_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    if len(text) <= limit:
        return text
    hidden = len(text) - limit
    return f"{text[:limit]}\n\n...[truncated {hidden} chars]"


def http_json_post(url: str, token: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except TimeoutError as err:
        raise RuntimeError(f"Network timeout after {timeout}s") from err
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code}: {detail}") from err
    except urllib.error.URLError as err:
        raise RuntimeError(f"Network error: {err}") from err

    try:
        return json.loads(raw)
    except json.JSONDecodeError as err:
        raise RuntimeError(f"Invalid JSON from API: {truncate(raw, 2000)}") from err


def normalize_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme:
        return "https://" + url
    return url


def html_to_text(raw_html: str) -> str:
    no_script = re.sub(r"<script[\s\S]*?</script>", " ", raw_html, flags=re.I)
    no_style = re.sub(r"<style[\s\S]*?</style>", " ", no_script, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", no_style)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ddg_extract_results(raw_html: str, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>[\s\S]*?)</a>',
        re.I,
    )
    for match in pattern.finditer(raw_html):
        href = html.unescape(match.group("href"))
        title_html = match.group("title")
        title = re.sub(r"<[^>]+>", "", title_html)
        title = html.unescape(title).strip()

        if "uddg=" in href:
            parsed = urllib.parse.urlparse(href)
            q = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
            if q:
                href = urllib.parse.unquote(q)

        if not href.startswith("http"):
            continue

        results.append({"title": title or "(no title)", "url": href})
        if len(results) >= max_results:
            break
    return results


class Agent:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        workspace: Path,
        thinking_enabled: bool,
        preserve_thinking: bool,
        max_tokens: int,
        temperature: float,
        shell_timeout: int,
        max_tool_steps: int,
        api_timeout: int,
        approval_mode: str,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.workspace = workspace.resolve()
        self.cwd = self.workspace
        self.thinking_enabled = thinking_enabled
        self.preserve_thinking = preserve_thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.shell_timeout = shell_timeout
        self.max_tool_steps = max_tool_steps
        self.api_timeout = api_timeout
        self.approval_mode = approval_mode
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _resolve_path(self, value: str, must_exist: bool = False) -> Path:
        raw = Path(value).expanduser()
        candidate = (self.cwd / raw).resolve() if not raw.is_absolute() else raw.resolve()
        try:
            candidate.relative_to(self.workspace)
        except ValueError as err:
            raise ValueError(f"Path is outside workspace: {candidate}") from err
        if must_exist and not candidate.exists():
            raise ValueError(f"Path does not exist: {candidate}")
        return candidate

    def _blocked_command(self, command: str) -> str | None:
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return pattern
        return None

    def _resolve_run_dir(self, value: str | None) -> Path:
        if value and value.strip():
            target = self._resolve_path(value.strip(), must_exist=True)
            if not target.is_dir():
                raise ValueError(f"Not a directory: {target}")
            return target
        return self.cwd

    def _run_exec(self, cmd: list[str], cwd: Path, timeout: int = DEFAULT_EXEC_TIMEOUT) -> dict[str, Any]:
        timeout = max(1, min(timeout, 600))
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "cwd": str(cwd),
                "command": cmd,
                "stdout": truncate(proc.stdout or ""),
                "stderr": truncate(proc.stderr or ""),
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"Command timed out after {timeout}s",
                "cwd": str(cwd),
                "command": cmd,
            }

    def _maybe_parse_json(self, value: str) -> Any | None:
        raw = value.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _requires_tool_approval(self, tool_name: str) -> bool:
        if self.approval_mode == "off":
            return False
        if self.approval_mode == "all":
            return True
        return tool_name == "shell"

    def _confirm_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        if not self._requires_tool_approval(tool_name):
            return True, None
        if not sys.stdin.isatty():
            return False, f"approval-mode={self.approval_mode} requires an interactive TTY"

        args_preview = truncate(json.dumps(arguments, ensure_ascii=False), 600)
        print(f"\nApproval required for tool `{tool_name}`", flush=True)
        print(args_preview, flush=True)
        try:
            answer = input("Ausfuehren? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False, "tool execution canceled during approval prompt"
        if answer not in {"y", "yes", "j", "ja"}:
            return False, "tool execution not approved by user"
        return True, None

    def _require_confirmation_phrase(
        self, arguments: dict[str, Any], expected: str, field: str = "confirm"
    ) -> tuple[bool, str | None]:
        value = str(arguments.get(field, "")).strip()
        if value != expected:
            return False, f"safety confirmation required: set `{field}` to `{expected}`"
        return True, None

    def _tool_shell(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = str(arguments.get("command", "")).strip()
        if not command:
            return {"ok": False, "error": "command is required"}

        bad_pattern = self._blocked_command(command)
        if bad_pattern:
            return {
                "ok": False,
                "error": f"Command blocked by safety policy (pattern: {bad_pattern})",
            }

        cwd_value = str(arguments.get("cwd", "")).strip()
        timeout = int(arguments.get("timeout_seconds", self.shell_timeout))
        timeout = max(1, min(timeout, 600))

        if cwd_value:
            try:
                run_cwd = self._resolve_path(cwd_value, must_exist=True)
            except ValueError as err:
                return {"ok": False, "error": str(err)}
        else:
            run_cwd = self.cwd

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(run_cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = truncate(proc.stdout or "")
            err = truncate(proc.stderr or "")
            return {
                "ok": proc.returncode == 0,
                "exit_code": proc.returncode,
                "cwd": str(run_cwd),
                "stdout": out,
                "stderr": err,
            }
        except subprocess.TimeoutExpired:
            return {
                "ok": False,
                "error": f"Command timed out after {timeout}s",
                "cwd": str(run_cwd),
            }

    def _tool_change_directory(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path", "")).strip()
        if not path_value:
            return {"ok": False, "error": "path is required"}
        try:
            target = self._resolve_path(path_value, must_exist=True)
        except ValueError as err:
            return {"ok": False, "error": str(err)}
        if not target.is_dir():
            return {"ok": False, "error": f"Not a directory: {target}"}
        self.cwd = target
        return {"ok": True, "cwd": str(self.cwd)}

    def _tool_list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path", ".")).strip() or "."
        max_entries = int(arguments.get("max_entries", 300))
        max_entries = max(1, min(max_entries, 2000))
        glob = str(arguments.get("glob", "")).strip()
        try:
            base = self._resolve_path(path_value, must_exist=True)
        except ValueError as err:
            return {"ok": False, "error": str(err)}
        if not base.is_dir():
            return {"ok": False, "error": f"Not a directory: {base}"}

        if shutil_which("rg"):
            cmd = ["rg", "--files", str(base)]
            if glob:
                cmd.extend(["--glob", glob])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            lines = lines[:max_entries]
            rel = [str(Path(p).resolve().relative_to(self.workspace)) for p in lines]
            return {"ok": True, "count": len(rel), "files": rel}

        files: list[str] = []
        for root, _, names in os.walk(base):
            for name in names:
                if len(files) >= max_entries:
                    break
                p = Path(root) / name
                rel = str(p.resolve().relative_to(self.workspace))
                if glob and not fnmatch_match(name, glob):
                    continue
                files.append(rel)
            if len(files) >= max_entries:
                break
        return {"ok": True, "count": len(files), "files": files}

    def _tool_read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path", "")).strip()
        if not path_value:
            return {"ok": False, "error": "path is required"}
        start_line = int(arguments.get("start_line", 1))
        end_line = int(arguments.get("end_line", 200))
        max_chars = int(arguments.get("max_chars", 12000))

        start_line = max(1, start_line)
        end_line = max(start_line, min(end_line, start_line + 2000))
        max_chars = max(200, min(max_chars, 50000))

        try:
            path = self._resolve_path(path_value, must_exist=True)
        except ValueError as err:
            return {"ok": False, "error": str(err)}
        if path.is_dir():
            return {"ok": False, "error": f"Path is a directory: {path}"}

        raw = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.splitlines()
        selected = lines[start_line - 1 : end_line]
        rendered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start_line))
        return {
            "ok": True,
            "path": str(path.relative_to(self.workspace)),
            "line_count": len(lines),
            "content": truncate(rendered, max_chars),
        }

    def _tool_write_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path", "")).strip()
        content = str(arguments.get("content", ""))
        mode = str(arguments.get("mode", "overwrite")).strip().lower()
        create_parents = bool(arguments.get("create_parents", True))

        if not path_value:
            return {"ok": False, "error": "path is required"}
        if mode not in {"overwrite", "append"}:
            return {"ok": False, "error": "mode must be overwrite or append"}

        try:
            path = self._resolve_path(path_value, must_exist=False)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        elif not path.parent.exists():
            return {"ok": False, "error": f"Parent directory does not exist: {path.parent}"}

        if mode == "overwrite":
            path.write_text(content, encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(content)

        return {
            "ok": True,
            "path": str(path.relative_to(self.workspace)),
            "bytes_written": len(content.encode("utf-8")),
            "mode": mode,
        }

    def _tool_replace_in_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path_value = str(arguments.get("path", "")).strip()
        old = str(arguments.get("old", ""))
        new = str(arguments.get("new", ""))
        replace_all = bool(arguments.get("replace_all", False))

        if not path_value:
            return {"ok": False, "error": "path is required"}
        if not old:
            return {"ok": False, "error": "old must be non-empty"}

        try:
            path = self._resolve_path(path_value, must_exist=True)
        except ValueError as err:
            return {"ok": False, "error": str(err)}
        if path.is_dir():
            return {"ok": False, "error": f"Path is a directory: {path}"}

        text = path.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            return {"ok": False, "error": "old text not found"}

        if replace_all:
            updated = text.replace(old, new)
            changed = occurrences
        else:
            updated = text.replace(old, new, 1)
            changed = 1

        path.write_text(updated, encoding="utf-8")
        return {
            "ok": True,
            "path": str(path.relative_to(self.workspace)),
            "replacements": changed,
        }

    def _tool_search_text(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = str(arguments.get("pattern", "")).strip()
        path_value = str(arguments.get("path", ".")).strip() or "."
        max_matches = int(arguments.get("max_matches", 200))
        max_matches = max(1, min(max_matches, 2000))

        if not pattern:
            return {"ok": False, "error": "pattern is required"}

        try:
            base = self._resolve_path(path_value, must_exist=True)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        if shutil_which("rg"):
            cmd = ["rg", "-n", "--hidden", "--glob", "!.git", pattern, str(base)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            return {
                "ok": True,
                "count": min(len(lines), max_matches),
                "matches": lines[:max_matches],
            }

        found: list[str] = []
        regex = re.compile(pattern)
        targets: list[Path] = []
        if base.is_file():
            targets.append(base)
        else:
            for root, _, files in os.walk(base):
                if ".git" in root.split(os.sep):
                    continue
                for name in files:
                    targets.append(Path(root) / name)

        for p in targets:
            if len(found) >= max_matches:
                break
            try:
                lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for idx, line in enumerate(lines, start=1):
                if regex.search(line):
                    rel = str(p.resolve().relative_to(self.workspace))
                    found.append(f"{rel}:{idx}:{line}")
                    if len(found) >= max_matches:
                        break
        return {"ok": True, "count": len(found), "matches": found}

    def _tool_web_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query", "")).strip()
        max_results = int(arguments.get("max_results", 5))
        max_results = max(1, min(max_results, 10))

        if not query:
            return {"ok": False, "error": "query is required"}

        url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; modal-glm5-agent/1.0)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as err:
            return {"ok": False, "error": f"web search failed: {err}"}

        items = ddg_extract_results(raw, max_results)
        return {
            "ok": True,
            "query": query,
            "count": len(items),
            "results": items,
        }

    def _tool_web_fetch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        url = str(arguments.get("url", "")).strip()
        max_chars = int(arguments.get("max_chars", 8000))
        max_chars = max(500, min(max_chars, 50000))

        if not url:
            return {"ok": False, "error": "url is required"}

        normalized = normalize_url(url)
        req = urllib.request.Request(
            normalized,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; modal-glm5-agent/1.0)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                ctype = resp.headers.get("Content-Type", "")
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as err:
            return {"ok": False, "error": f"web fetch failed: {err}", "url": normalized}

        title_match = re.search(r"<title[^>]*>([\s\S]*?)</title>", raw, re.I)
        title = html.unescape(title_match.group(1).strip()) if title_match else ""
        text = html_to_text(raw)

        return {
            "ok": True,
            "url": normalized,
            "content_type": ctype,
            "title": title,
            "content": truncate(text, max_chars),
        }

    def _tool_get_time(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _ = arguments
        now = datetime.now(timezone.utc)
        return {
            "ok": True,
            "utc": now.isoformat(),
            "unix": int(now.timestamp()),
        }

    def _tool_git_status(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("git"):
            return {"ok": False, "error": "git is not installed or not in PATH"}
        path_value = str(arguments.get("path", ".")).strip()
        short = bool(arguments.get("short", True))
        branch = bool(arguments.get("branch", True))
        timeout = int(arguments.get("timeout_seconds", 30))
        try:
            run_dir = self._resolve_run_dir(path_value)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        cmd = ["git", "status"]
        if short:
            cmd.append("--short")
        if branch:
            cmd.append("--branch")
        return self._run_exec(cmd, cwd=run_dir, timeout=timeout)

    def _tool_git_diff(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("git"):
            return {"ok": False, "error": "git is not installed or not in PATH"}
        path_value = str(arguments.get("path", ".")).strip()
        staged = bool(arguments.get("staged", False))
        target = str(arguments.get("target", "")).strip()
        file_path = str(arguments.get("file", "")).strip()
        timeout = int(arguments.get("timeout_seconds", 60))
        try:
            run_dir = self._resolve_run_dir(path_value)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if target:
            cmd.append(target)
        if file_path:
            cmd.extend(["--", file_path])
        return self._run_exec(cmd, cwd=run_dir, timeout=timeout)

    def _tool_git_add(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("git"):
            return {"ok": False, "error": "git is not installed or not in PATH"}
        path_value = str(arguments.get("path", ".")).strip()
        timeout = int(arguments.get("timeout_seconds", 30))
        raw_paths = arguments.get("paths", ["."])
        if isinstance(raw_paths, str):
            paths = [raw_paths]
        elif isinstance(raw_paths, list):
            paths = [str(p) for p in raw_paths if str(p).strip()]
        else:
            return {"ok": False, "error": "paths must be a string or array of strings"}
        if not paths:
            paths = ["."]
        try:
            run_dir = self._resolve_run_dir(path_value)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        cmd = ["git", "add", "--", *paths]
        return self._run_exec(cmd, cwd=run_dir, timeout=timeout)

    def _tool_git_commit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("git"):
            return {"ok": False, "error": "git is not installed or not in PATH"}
        message = str(arguments.get("message", "")).strip()
        if not message:
            return {"ok": False, "error": "message is required"}
        path_value = str(arguments.get("path", ".")).strip()
        timeout = int(arguments.get("timeout_seconds", 60))
        all_tracked = bool(arguments.get("all", False))
        no_verify = bool(arguments.get("no_verify", False))
        allow_empty = bool(arguments.get("allow_empty", False))
        try:
            run_dir = self._resolve_run_dir(path_value)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        cmd = ["git", "commit"]
        if all_tracked:
            cmd.append("-a")
        if no_verify:
            cmd.append("--no-verify")
        if allow_empty:
            cmd.append("--allow-empty")
        cmd.extend(["-m", message])
        return self._run_exec(cmd, cwd=run_dir, timeout=timeout)

    def _tool_git_log(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("git"):
            return {"ok": False, "error": "git is not installed or not in PATH"}
        path_value = str(arguments.get("path", ".")).strip()
        limit = int(arguments.get("limit", 10))
        timeout = int(arguments.get("timeout_seconds", 30))
        limit = max(1, min(limit, 50))
        try:
            run_dir = self._resolve_run_dir(path_value)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        cmd = ["git", "log", f"-n{limit}", "--oneline", "--decorate"]
        return self._run_exec(cmd, cwd=run_dir, timeout=timeout)

    def _tool_git_push(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("git"):
            return {"ok": False, "error": "git is not installed or not in PATH"}
        path_value = str(arguments.get("path", ".")).strip()
        remote = str(arguments.get("remote", "origin")).strip() or "origin"
        branch = str(arguments.get("branch", "")).strip()
        set_upstream = bool(arguments.get("set_upstream", False))
        force_with_lease = bool(arguments.get("force_with_lease", False))
        no_verify = bool(arguments.get("no_verify", False))
        dry_run = bool(arguments.get("dry_run", True))
        timeout = int(arguments.get("timeout_seconds", 90))

        if set_upstream and not branch:
            return {"ok": False, "error": "branch is required when set_upstream=true"}
        if not dry_run:
            ok, err = self._require_confirmation_phrase(arguments, expected="PUSH")
            if not ok:
                return {"ok": False, "error": err}

        try:
            run_dir = self._resolve_run_dir(path_value)
        except ValueError as err:
            return {"ok": False, "error": str(err)}

        cmd = ["git", "push"]
        if dry_run:
            cmd.append("--dry-run")
        if set_upstream:
            cmd.append("--set-upstream")
        if force_with_lease:
            cmd.append("--force-with-lease")
        if no_verify:
            cmd.append("--no-verify")
        cmd.append(remote)
        if branch:
            cmd.append(branch)

        result = self._run_exec(cmd, cwd=run_dir, timeout=timeout)
        result["safety"] = {
            "dry_run": dry_run,
            "confirmation_required_if_not_dry_run": "PUSH",
        }
        return result

    def _tool_gh_pr_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("gh"):
            return {"ok": False, "error": "gh CLI is not installed or not in PATH"}
        repo = str(arguments.get("repo", "")).strip()
        state = str(arguments.get("state", "open")).strip()
        limit = int(arguments.get("limit", 20))
        timeout = int(arguments.get("timeout_seconds", 45))
        if state not in {"open", "closed", "merged", "all"}:
            return {"ok": False, "error": "state must be one of: open, closed, merged, all"}
        limit = max(1, min(limit, 100))

        cmd = [
            "gh",
            "pr",
            "list",
            "--state",
            state,
            "--limit",
            str(limit),
            "--json",
            "number,title,url,headRefName,baseRefName,author,updatedAt,isDraft",
        ]
        if repo:
            cmd.extend(["--repo", repo])
        result = self._run_exec(cmd, cwd=self.cwd, timeout=timeout)
        parsed = self._maybe_parse_json(result.get("stdout", ""))
        if parsed is not None:
            result["data"] = parsed
        return result

    def _tool_gh_pr_view(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("gh"):
            return {"ok": False, "error": "gh CLI is not installed or not in PATH"}
        repo = str(arguments.get("repo", "")).strip()
        pr_number = arguments.get("pr_number")
        timeout = int(arguments.get("timeout_seconds", 45))

        cmd = [
            "gh",
            "pr",
            "view",
            "--json",
            "number,title,body,url,state,headRefName,baseRefName,author,mergeStateStatus,isDraft,reviews,comments",
        ]
        if pr_number is not None and str(pr_number).strip():
            cmd.insert(3, str(pr_number))
        if repo:
            cmd.extend(["--repo", repo])
        result = self._run_exec(cmd, cwd=self.cwd, timeout=timeout)
        parsed = self._maybe_parse_json(result.get("stdout", ""))
        if parsed is not None:
            result["data"] = parsed
        return result

    def _tool_gh_pr_create(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("gh"):
            return {"ok": False, "error": "gh CLI is not installed or not in PATH"}
        title = str(arguments.get("title", "")).strip()
        body = str(arguments.get("body", "")).strip()
        if not title or not body:
            return {"ok": False, "error": "title and body are required"}
        repo = str(arguments.get("repo", "")).strip()
        base = str(arguments.get("base", "")).strip()
        head = str(arguments.get("head", "")).strip()
        draft = bool(arguments.get("draft", False))
        timeout = int(arguments.get("timeout_seconds", 90))

        cmd = ["gh", "pr", "create", "--title", title, "--body", body]
        if base:
            cmd.extend(["--base", base])
        if head:
            cmd.extend(["--head", head])
        if draft:
            cmd.append("--draft")
        if repo:
            cmd.extend(["--repo", repo])
        return self._run_exec(cmd, cwd=self.cwd, timeout=timeout)

    def _tool_gh_pr_comment(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("gh"):
            return {"ok": False, "error": "gh CLI is not installed or not in PATH"}
        pr_number = arguments.get("pr_number")
        body = str(arguments.get("body", "")).strip()
        if pr_number is None or not str(pr_number).strip():
            return {"ok": False, "error": "pr_number is required"}
        if not body:
            return {"ok": False, "error": "body is required"}
        repo = str(arguments.get("repo", "")).strip()
        timeout = int(arguments.get("timeout_seconds", 45))

        cmd = ["gh", "pr", "comment", str(pr_number), "--body", body]
        if repo:
            cmd.extend(["--repo", repo])
        return self._run_exec(cmd, cwd=self.cwd, timeout=timeout)

    def _tool_gh_pr_merge(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if not shutil_which("gh"):
            return {"ok": False, "error": "gh CLI is not installed or not in PATH"}
        repo = str(arguments.get("repo", "")).strip()
        pr_number = arguments.get("pr_number")
        merge_method = str(arguments.get("merge_method", "squash")).strip().lower()
        delete_branch = bool(arguments.get("delete_branch", True))
        admin = bool(arguments.get("admin", False))
        auto = bool(arguments.get("auto", False))
        execute = bool(arguments.get("execute", False))
        match_head_commit = str(arguments.get("match_head_commit", "")).strip()
        timeout = int(arguments.get("timeout_seconds", 90))

        if merge_method not in {"merge", "squash", "rebase"}:
            return {"ok": False, "error": "merge_method must be one of: merge, squash, rebase"}

        pr_ref = str(pr_number).strip() if pr_number is not None and str(pr_number).strip() else "CURRENT"
        expected_confirm = f"MERGE-{pr_ref}"

        cmd = ["gh", "pr", "merge"]
        if pr_number is not None and str(pr_number).strip():
            cmd.append(str(pr_number))
        if merge_method == "merge":
            cmd.append("--merge")
        elif merge_method == "rebase":
            cmd.append("--rebase")
        else:
            cmd.append("--squash")
        if delete_branch:
            cmd.append("--delete-branch")
        if admin:
            cmd.append("--admin")
        if auto:
            cmd.append("--auto")
        if match_head_commit:
            cmd.extend(["--match-head-commit", match_head_commit])
        if repo:
            cmd.extend(["--repo", repo])

        if not execute:
            return {
                "ok": True,
                "preview_only": True,
                "would_run": cmd,
                "safety": {
                    "execute": False,
                    "required_for_execute": {
                        "execute": True,
                        "confirm": expected_confirm,
                    },
                },
            }

        ok, err = self._require_confirmation_phrase(arguments, expected=expected_confirm)
        if not ok:
            return {"ok": False, "error": err, "expected_confirm": expected_confirm}

        result = self._run_exec(cmd, cwd=self.cwd, timeout=timeout)
        result["safety"] = {
            "execute": True,
            "confirm_used": expected_confirm,
        }
        return result

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a shell command in the current workspace or a child directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Command to execute."},
                            "cwd": {
                                "type": "string",
                                "description": "Optional directory relative to workspace.",
                            },
                            "timeout_seconds": {
                                "type": "integer",
                                "description": "Timeout in seconds (1-600).",
                                "minimum": 1,
                                "maximum": 600,
                            },
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "change_directory",
                    "description": "Change current working directory for future shell commands.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path."}
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files under a directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path."},
                            "glob": {
                                "type": "string",
                                "description": "Optional rg glob filter, e.g. *.py",
                            },
                            "max_entries": {"type": "integer", "minimum": 1, "maximum": 2000},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file with line numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                            "max_chars": {"type": "integer", "minimum": 200},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file (overwrite or append).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "mode": {"type": "string", "enum": ["overwrite", "append"]},
                            "create_parents": {"type": "boolean"},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "replace_in_file",
                    "description": "Replace text in a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old": {"type": "string"},
                            "new": {"type": "string"},
                            "replace_all": {"type": "boolean"},
                        },
                        "required": ["path", "old", "new"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_text",
                    "description": "Search text in files using ripgrep.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"},
                            "max_matches": {"type": "integer", "minimum": 1, "maximum": 2000},
                        },
                        "required": ["pattern"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for recent or factual information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_fetch",
                    "description": "Fetch and extract readable text from a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "max_chars": {"type": "integer", "minimum": 500, "maximum": 50000},
                        },
                        "required": ["url"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Return current UTC time.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_status",
                    "description": "Show git status for a repository directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "short": {"type": "boolean"},
                            "branch": {"type": "boolean"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_diff",
                    "description": "Show git diff (optionally staged or for a specific file).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "staged": {"type": "boolean"},
                            "target": {"type": "string"},
                            "file": {"type": "string"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_add",
                    "description": "Stage files in git.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "paths": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ]
                            },
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_commit",
                    "description": "Create a git commit with a message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "message": {"type": "string"},
                            "all": {"type": "boolean"},
                            "no_verify": {"type": "boolean"},
                            "allow_empty": {"type": "boolean"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_log",
                    "description": "Show recent git commits.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "git_push",
                    "description": "Push commits to remote. Safe by default with dry_run=true.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "remote": {"type": "string"},
                            "branch": {"type": "string"},
                            "set_upstream": {"type": "boolean"},
                            "force_with_lease": {"type": "boolean"},
                            "no_verify": {"type": "boolean"},
                            "dry_run": {"type": "boolean"},
                            "confirm": {
                                "type": "string",
                                "description": "Required value `PUSH` when dry_run=false.",
                            },
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "gh_pr_list",
                    "description": "List pull requests with GitHub CLI.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "owner/repo (optional)"},
                            "state": {"type": "string", "enum": ["open", "closed", "merged", "all"]},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "gh_pr_view",
                    "description": "View pull request details with GitHub CLI.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "owner/repo (optional)"},
                            "pr_number": {"type": "integer"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "gh_pr_create",
                    "description": "Create a pull request with GitHub CLI.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "owner/repo (optional)"},
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "base": {"type": "string"},
                            "head": {"type": "string"},
                            "draft": {"type": "boolean"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                        "required": ["title", "body"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "gh_pr_comment",
                    "description": "Comment on a pull request with GitHub CLI.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "owner/repo (optional)"},
                            "pr_number": {"type": "integer"},
                            "body": {"type": "string"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                        "required": ["pr_number", "body"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "gh_pr_merge",
                    "description": "Merge a PR with explicit safety confirmation. Preview by default.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo": {"type": "string", "description": "owner/repo (optional)"},
                            "pr_number": {"type": "integer"},
                            "merge_method": {"type": "string", "enum": ["merge", "squash", "rebase"]},
                            "delete_branch": {"type": "boolean"},
                            "admin": {"type": "boolean"},
                            "auto": {"type": "boolean"},
                            "match_head_commit": {"type": "string"},
                            "execute": {
                                "type": "boolean",
                                "description": "Set true to run merge. Otherwise preview only.",
                            },
                            "confirm": {
                                "type": "string",
                                "description": "Required when execute=true: `MERGE-<PR_NUMBER>` or `MERGE-CURRENT`.",
                            },
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600},
                        },
                    },
                },
            },
        ]

    def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "shell":
            return self._tool_shell(arguments)
        if name == "change_directory":
            return self._tool_change_directory(arguments)
        if name == "list_files":
            return self._tool_list_files(arguments)
        if name == "read_file":
            return self._tool_read_file(arguments)
        if name == "write_file":
            return self._tool_write_file(arguments)
        if name == "replace_in_file":
            return self._tool_replace_in_file(arguments)
        if name == "search_text":
            return self._tool_search_text(arguments)
        if name == "web_search":
            return self._tool_web_search(arguments)
        if name == "web_fetch":
            return self._tool_web_fetch(arguments)
        if name == "get_time":
            return self._tool_get_time(arguments)
        if name == "git_status":
            return self._tool_git_status(arguments)
        if name == "git_diff":
            return self._tool_git_diff(arguments)
        if name == "git_add":
            return self._tool_git_add(arguments)
        if name == "git_commit":
            return self._tool_git_commit(arguments)
        if name == "git_log":
            return self._tool_git_log(arguments)
        if name == "git_push":
            return self._tool_git_push(arguments)
        if name == "gh_pr_list":
            return self._tool_gh_pr_list(arguments)
        if name == "gh_pr_view":
            return self._tool_gh_pr_view(arguments)
        if name == "gh_pr_create":
            return self._tool_gh_pr_create(arguments)
        if name == "gh_pr_comment":
            return self._tool_gh_pr_comment(arguments)
        if name == "gh_pr_merge":
            return self._tool_gh_pr_merge(arguments)
        return {"ok": False, "error": f"Unknown tool: {name}"}

    def _call_model(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "tools": self.tool_schemas(),
            "tool_choice": "auto",
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "thinking": {
                "type": "enabled" if self.thinking_enabled else "disabled",
                "clear_thinking": not self.preserve_thinking,
            },
        }
        return http_json_post(
            f"{self.base_url}/chat/completions",
            token=self.api_key,
            payload=payload,
            timeout=self.api_timeout,
        )

    def _format_tool_brief(self, tc: dict[str, Any]) -> str:
        name = tc.get("function", {}).get("name", "unknown")
        args = tc.get("function", {}).get("arguments", "{}")
        if isinstance(args, str):
            preview = truncate(args, 200)
        else:
            preview = truncate(json.dumps(args, ensure_ascii=False), 200)
        return f"{name}({preview})"

    def run_turn(self, user_input: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        tool_fingerprints: list[str] = []

        for step in range(1, self.max_tool_steps + 1):
            response = self._call_model()
            choices = response.get("choices") or []
            if not choices:
                print("Fehler: keine choices im Modell-Response", flush=True)
                print(json.dumps(response, indent=2, ensure_ascii=False), flush=True)
                return

            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            tool_calls = message.get("tool_calls") or []

            if reasoning and self.thinking_enabled:
                print("\nDenken:\n" + reasoning, flush=True)

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            if reasoning and self.preserve_thinking:
                assistant_msg["reasoning_content"] = reasoning
            self.messages.append(assistant_msg)

            if not tool_calls:
                print("\nAntwort:\n" + (content if content.strip() else "(leer)"), flush=True)
                return

            fingerprint = json.dumps(
                [{"name": tc.get("function", {}).get("name"), "args": tc.get("function", {}).get("arguments")} for tc in tool_calls],
                ensure_ascii=False,
                sort_keys=True,
            )
            tool_fingerprints.append(fingerprint)
            if len(tool_fingerprints) >= 3 and tool_fingerprints[-1] == tool_fingerprints[-2] == tool_fingerprints[-3]:
                print(
                    "\nAntwort:\nAbbruch: identische Tool-Calls wurden 3x wiederholt. "
                    "Bitte Anfrage praezisieren oder --max-tool-steps anpassen.",
                    flush=True,
                )
                return

            print(f"\nTools ({len(tool_calls)}) [step {step}/{self.max_tool_steps}]:", flush=True)
            for tc in tool_calls:
                print(f"- {self._format_tool_brief(tc)}", flush=True)

            for tc in tool_calls:
                call_id = tc.get("id", "")
                fn = tc.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})

                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                else:
                    parsed_args = {}

                approved, deny_reason = self._confirm_tool_call(name, parsed_args)
                if not approved:
                    result = {"ok": False, "error": deny_reason or "tool execution denied", "tool": name}
                else:
                    result = self._dispatch_tool(name, parsed_args)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                self.messages.append(tool_message)

        print(
            "\nAntwort:\n"
            "Max tool steps reached. Please refine the request or increase --max-tool-steps.",
            flush=True,
        )


def shutil_which(cmd: str) -> str | None:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        full = Path(path) / cmd
        if full.exists() and os.access(full, os.X_OK):
            return str(full)
    return None


def fnmatch_match(name: str, pattern: str) -> bool:
    regex = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.fullmatch(regex, name) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLM-5 local coding + research agent (Modal endpoint)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prompt", nargs="*", help="Optional one-shot prompt. If empty, starts REPL.")
    parser.add_argument("--workspace", default=str(Path.home()), help="Workspace root boundary.")
    parser.add_argument("--model", default=None, help="Model id override.")
    parser.add_argument("--base-url", default=None, help="API base URL override.")
    parser.add_argument("--no-thinking", action="store_true", help="Disable reasoning output.")
    parser.add_argument(
        "--preserve-thinking",
        action="store_true",
        help="Preserve reasoning_content across turns (for coding agents).",
    )
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max output tokens per call.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature 0..1.")
    parser.add_argument("--shell-timeout", type=int, default=DEFAULT_SHELL_TIMEOUT, help="Shell timeout.")
    parser.add_argument("--max-tool-steps", type=int, default=12, help="Max model-tool loops per turn.")
    parser.add_argument("--api-timeout", type=int, default=180, help="API request timeout in seconds.")
    parser.add_argument(
        "--approval-mode",
        choices=["off", "shell", "all"],
        default="off",
        help="Tool approval mode: off, shell-only, or all tools.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    env_values = load_env_file(script_dir / ".env")

    api_key = os.environ.get("MODAL_API_KEY") or env_values.get("MODAL_API_KEY")
    if not api_key:
        print("Fehler: MODAL_API_KEY fehlt. Starte zuerst ./setup.sh oder export MODAL_API_KEY.")
        return 1

    base_url = (
        args.base_url
        or os.environ.get("MODAL_BASE_URL")
        or env_values.get("MODAL_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = args.model or os.environ.get("MODEL_ID") or env_values.get("MODEL_ID") or DEFAULT_MODEL

    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        print(f"Fehler: workspace ist kein Verzeichnis: {workspace}")
        return 1

    agent = Agent(
        api_key=api_key,
        base_url=base_url,
        model=model,
        workspace=workspace,
        thinking_enabled=not args.no_thinking,
        preserve_thinking=args.preserve_thinking,
        max_tokens=max(1, min(args.max_tokens, 131072)),
        temperature=max(0.0, min(args.temperature, 1.0)),
        shell_timeout=max(1, min(args.shell_timeout, 600)),
        max_tool_steps=max(1, min(args.max_tool_steps, 40)),
        api_timeout=max(10, min(args.api_timeout, 600)),
        approval_mode=args.approval_mode,
    )

    if args.prompt:
        agent.run_turn(" ".join(args.prompt))
        return 0

    print("GLM-5 Agent REPL")
    print(f"workspace: {agent.workspace}")
    print(f"model: {agent.model}")
    print(f"approval-mode: {agent.approval_mode}")
    print("commands: /reset, /cwd, /quit")

    while True:
        try:
            user_input = input("\nDu> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_input:
            continue
        if user_input in {"/quit", "/exit"}:
            print("Bye.")
            return 0
        if user_input == "/reset":
            agent.reset()
            print("Kontext zurueckgesetzt.")
            continue
        if user_input == "/cwd":
            print(f"cwd: {agent.cwd}")
            continue

        try:
            agent.run_turn(user_input)
        except Exception as err:
            print(f"Fehler: {err}")


if __name__ == "__main__":
    raise SystemExit(main())
