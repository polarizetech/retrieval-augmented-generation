"""Set the gateway URL in the scientific-research-rag plugin's MCP configs.

Usage: python scripts/configure_plugin.py [URL]

The URL comes from the argument, else the MCP_PUBLIC_URL environment variable, else an
MCP_PUBLIC_URL= line in config/rag.env. A trailing slash is stripped. The URL must start with
https:// and must not contain "<" (the placeholder). Both files in plugin/scientific-research-rag
(.mcp.json and mcp.json) are rewritten with the URL.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = ROOT / "plugin" / "scientific-research-rag"
PLUGIN_FILES = (PLUGIN_DIR / ".mcp.json", PLUGIN_DIR / "mcp.json")
ENV_FILE = ROOT / "config" / "rag.env"
SERVER = "scientific-research-rag"


def url_from_env_file(path: Path) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MCP_PUBLIC_URL="):
            return line.split("=", 1)[1].strip()
    return ""


def resolve_url(argv: list[str]) -> str:
    if argv:
        return argv[0].strip()
    return (os.environ.get("MCP_PUBLIC_URL") or url_from_env_file(ENV_FILE)).strip()


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    url = resolve_url(args).rstrip("/")
    if not url:
        print(
            "configure_plugin: no URL given (argument, MCP_PUBLIC_URL or config/rag.env).",
            file=sys.stderr,
        )
        return 2
    if not url.startswith("https://"):
        print("configure_plugin: the URL must start with https://.", file=sys.stderr)
        return 2
    if "<" in url:
        print("configure_plugin: the URL still contains a placeholder ('<').", file=sys.stderr)
        return 2
    for path in PLUGIN_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["mcpServers"][SERVER]["url"] = url
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"updated {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
