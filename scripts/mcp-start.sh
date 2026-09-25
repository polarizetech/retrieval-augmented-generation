#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
set -a; source "$ROOT/config/rag.env"; set +a
[[ "$MCP_CONFIG" = /* ]] || MCP_CONFIG="$ROOT/$MCP_CONFIG"
[[ "$MCP_TOKEN_FILE" = /* ]] || MCP_TOKEN_FILE="$ROOT/$MCP_TOKEN_FILE"
[ -f "$MCP_CONFIG" ] || { echo "Missing $MCP_CONFIG" >&2; exit 1; }

# The gateway fronts the paper library and any write tools its upstreams expose. A non-loopback
# bind puts them on every reachable interface, so it must be an explicit, deliberate choice. Put
# an authenticating proxy in front of loopback instead of widening the bind.
case "${MCP_HOST:-127.0.0.1}" in
  127.0.0.1|localhost|::1) ;;
  *)
    if [ "${MCP_ALLOW_PUBLIC_BIND:-0}" != "1" ]; then
      echo "Refusing to bind MCP gateway to $MCP_HOST." >&2
      echo "Set MCP_HOST=127.0.0.1, or re-run with MCP_ALLOW_PUBLIC_BIND=1 to override." >&2
      exit 1
    fi
    echo "WARNING: binding MCP gateway to $MCP_HOST (MCP_ALLOW_PUBLIC_BIND=1)." >&2
    ;;
esac
mkdir -p "$ROOT/logs" "$ROOT/run"
pidfile="$ROOT/run/mcp.pid"
children_pidfile="$ROOT/run/mcp-children.pid"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "MCP gateway already running (pid $(cat "$pidfile"))"
  "$HERE/mcp-url.sh"
  exit 0
fi
# The previous gateway process is not running. If it crashed hard (SIGKILL/OOM) instead of
# shutting down cleanly, its upstream stdio child processes (the paper library and any other
# upstreams) can be left running as orphans. Reap anything the last run recorded.
if [ -f "$children_pidfile" ]; then
  reaped=0
  while read -r cpid; do
    if [ -n "$cpid" ] && kill -0 "$cpid" 2>/dev/null; then
      kill "$cpid" 2>/dev/null && reaped=$((reaped + 1))
    fi
  done < "$children_pidfile"
  rm -f "$children_pidfile"
  [ "$reaped" -gt 0 ] && echo "Reaped $reaped orphaned MCP upstream process(es) from a previous crash"
fi
nohup "$ROOT/.venv/bin/research-rag-gateway" --http \
  --config "$MCP_CONFIG" --host "$MCP_HOST" --port "$MCP_PORT" --token-file "$MCP_TOKEN_FILE" \
  >"$ROOT/logs/mcp.log" 2>&1 &
echo $! >"$pidfile"
sleep 2
if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  echo "MCP gateway failed; see $ROOT/logs/mcp.log" >&2
  exit 1
fi
echo "MCP gateway started (pid $(cat "$pidfile"))"
"$HERE/mcp-url.sh"
