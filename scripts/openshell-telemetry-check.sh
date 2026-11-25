#!/bin/sh
# Fails unless OpenShell telemetry is off on this machine (see docs/openshell.md).
# Checks the gateway env file, the running gateway's environment, and open connections.
set -u

env_file="${XDG_CONFIG_HOME:-$HOME/.config}/openshell/gateway.env"
telemetry_host="events.telemetry.data.nvidia.com"
fail=0

bad() {
  echo "FAIL: $1"
  fail=1
}

ok() {
  echo "ok:   $1"
}

if [ ! -f "$env_file" ]; then
  bad "$env_file is missing; the gateway would start with telemetry ON"
else
  if grep -Eq '^[[:space:]]*(export[[:space:]]+)?OPENSHELL_TELEMETRY_ENABLED=["'\'']?(false|0|no|off)["'\'']?[[:space:]]*$' "$env_file"; then
    ok "OPENSHELL_TELEMETRY_ENABLED=false in $env_file"
  else
    bad "$env_file does not set OPENSHELL_TELEMETRY_ENABLED=false"
  fi
  if grep -Eq '^[[:space:]]*(export[[:space:]]+)?OPENSHELL_TELEMETRY_ENDPOINT=["'\'']?http://127\.0\.0\.1:' "$env_file"; then
    ok "telemetry endpoint points at a dead local port"
  else
    bad "$env_file does not point OPENSHELL_TELEMETRY_ENDPOINT at 127.0.0.1"
  fi
fi

pids=$(pgrep -x openshell-gateway 2>/dev/null || true)
if [ -z "$pids" ]; then
  echo "info: no openshell-gateway process is running; only the config was checked"
else
  for pid in $pids; do
    # `ps eww` shows a process's environment for processes owned by the same user.
    if ps eww -p "$pid" -o command= 2>/dev/null | tr ' ' '\n' | grep -qx 'OPENSHELL_TELEMETRY_ENABLED=false'; then
      ok "running gateway $pid has OPENSHELL_TELEMETRY_ENABLED=false"
    else
      bad "running gateway $pid lacks OPENSHELL_TELEMETRY_ENABLED=false"
    fi
  done
fi

addrs=$(dig +short "$telemetry_host" 2>/dev/null | grep -E '^[0-9.]+$' || true)
procs=$(pgrep -f 'openshell' 2>/dev/null || true)
if [ -n "$addrs" ] && [ -n "$procs" ]; then
  conns=$(for pid in $procs; do lsof -nP -a -p "$pid" -iTCP 2>/dev/null; done)
  open=""
  for addr in $addrs; do
    if printf '%s\n' "$conns" | grep -qF -- "->$addr:"; then
      open="$addr"
    fi
  done
  if [ -n "$open" ]; then
    bad "an OpenShell process has a connection open to $telemetry_host"
  else
    ok "no OpenShell process is connected to $telemetry_host"
  fi
fi

exit "$fail"
