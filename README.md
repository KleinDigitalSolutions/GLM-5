# Modal GLM-5 Setup

## 1) API Key sicher speichern

```bash
cd ~/modal-glm5
chmod +x setup.sh chat.sh agent.sh
./setup.sh
```

## 2) Quick test (simple chat)

```bash
./chat.sh "How many r-s are in strawberry?"
```

## 3) Full coding/research agent

One-shot:

```bash
./agent.sh --workspace ~/ "Analysiere ~/the-forge-community und finde TODOs"
```

Interactive REPL:

```bash
./agent.sh --workspace ~/
```

Mit Shell-Bestaetigung vor jedem Shell-Toolcall:

```bash
./agent.sh --workspace ~/ --approval-mode shell
```

REPL commands:

- `/reset` reset conversation context
- `/cwd` show current agent working directory
- `/quit` exit

Useful flags:

- `--no-thinking` disable reasoning output
- `--preserve-thinking` keep reasoning context across tool loops
- `--max-tool-steps 20` allow longer tool planning loops
- `--shell-timeout 180` increase command timeout
- `--api-timeout 240` increase single API call timeout
- `--approval-mode shell` require approval before each `shell` tool call
- `--approval-mode all` require approval for every tool call
- `--model zai-org/GLM-5-FP8` model override

## Built-in tools in `agent.py`

- `shell`: run commands
- `change_directory`: set agent CWD
- `list_files`: enumerate files (uses `rg --files` if available)
- `read_file`: read file ranges with line numbers
- `write_file`: create/overwrite/append files
- `replace_in_file`: patch text by replacement
- `search_text`: grep/ripgrep-like search
- `web_search`: web search (DuckDuckGo HTML endpoint)
- `web_fetch`: fetch and extract readable content from URL
- `get_time`: current UTC timestamp
- `git_status`: repository status
- `git_diff`: unstaged/staged diffs
- `git_add`: stage files
- `git_commit`: create commits
- `git_log`: recent commits
- `git_push`: push to remote (default `dry_run=true`; execution needs `confirm: "PUSH"`)
- `gh_pr_list`: list pull requests
- `gh_pr_view`: inspect a pull request
- `gh_pr_create`: create pull requests
- `gh_pr_comment`: comment on pull requests
- `gh_pr_merge`: merge pull requests (preview by default; execute requires confirm phrase)

## Safety defaults

- Workspace boundary: file tools are restricted to `--workspace`.
- Command guard: blocks obvious destructive/admin commands (`sudo`, `shutdown`, `rm -rf /`, etc.).
- Command timeout and output truncation are enabled by default.
- `approval-mode` can force manual confirmation per tool call.
- `git_push` requires confirm phrase `PUSH` when `dry_run=false`.
- `gh_pr_merge` runs preview by default; execute needs `confirm` like `MERGE-123`.

## Notes

- API key is stored in `~/modal-glm5/.env` with permission `600`.
- `chat.sh` uses `jq` for JSON parsing/output.
- If `jq` is missing: `brew install jq`
- For `gh_*` tools, authenticate once with: `gh auth login`

## Push/Merge safety flow

Typical safe flow:

1. Ask agent for preview with `git_push` (default dry-run) or `gh_pr_merge` (`execute=false`).
2. Review command/results.
3. Execute explicitly with confirmation phrase:
- `git_push`: set `dry_run=false` and `confirm="PUSH"`
- `gh_pr_merge`: set `execute=true` and `confirm="MERGE-<PR_NUMBER>"`

## Docs used for implementation

- https://modal.com/blog/try-glm-5
- https://docs.z.ai/guides/llm/glm-5
- https://docs.z.ai/guides/capabilities/function-calling
- https://docs.z.ai/guides/capabilities/thinking
- https://docs.z.ai/guides/overview/concept-param
