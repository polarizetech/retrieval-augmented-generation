#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
pidfile="$ROOT/run/mcp.pid"
children_pidfile="$ROOT/run/mcp-children.pid"
if [ ! -f "$pidfile" ]; then
  echo "MCP gateway is not running"
  exit 0
fi
pid="$(cat "$pidfile")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped MCP gateway (pid $pid)"
fi
rm -f "$pidfile"
# A clean shutdown already terminates upstream children via the gateway's own lifespan
# cleanup and clears this file; remove it defensively in case that step was interrupted.
if [ -f "$children_pidfile" ]; then
  while read -r cpid; do
    [ -n "$cpid" ] && kill -0 "$cpid" 2>/dev/null && kill "$cpid" 2>/dev/null
  done < "$children_pidfile"
  rm -f "$children_pidfile"
fi
