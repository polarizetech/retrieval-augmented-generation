#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
set -a; source "$ROOT/config/rag.env"; set +a
[[ "$MCP_TOKEN_FILE" = /* ]] || MCP_TOKEN_FILE="$ROOT/$MCP_TOKEN_FILE"
[ -f "$MCP_TOKEN_FILE" ] || { echo "MCP token not created yet; start the gateway first." >&2; exit 1; }
token="$(cat "$MCP_TOKEN_FILE")"
echo "Local host:  http://127.0.0.1:$MCP_PORT/$token/mcp"
echo "Keep this URL private: the path token grants access to the MCP server."
