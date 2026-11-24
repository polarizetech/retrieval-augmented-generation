#!/usr/bin/env bash
# Runs inside the cloud image: one batch of questions, then exits (see docs/cloud.md).
#
# Expects: /batch/questions.txt (one question per line; blank lines and # comments skipped),
# BATCH_ID, the secrets in the environment (PIPELINE_ENDPOINT_KEY, PAPER_FETCH_S3_*), and
# optionally the private agent briefs mounted at /agents/private.
# Leaves: /work/runs/<run>/ for each question, and /work/batch.json, a summary.
set -uo pipefail

BATCH_ID="${BATCH_ID:?BATCH_ID is required}"
PREFIX="${CLOUD_PREFIX:-pipeline-cloud}"
QUESTIONS=/batch/questions.txt
SYNC="python /opt/rag/cloud/index_sync.py"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

[ -f "$QUESTIONS" ] || { log "no $QUESTIONS"; exit 2; }
[ -n "${PIPELINE_ENDPOINT_KEY:-}" ] || { log "PIPELINE_ENDPOINT_KEY is not set"; exit 2; }
[ -d /agents/private ] || unset PIPELINE_AGENT_EXTENSIONS_DIR
mkdir -p /work/runs /work/data /work/index

log "starting ollama"
ollama serve >/work/ollama.log 2>&1 &
for _ in $(seq 60); do ollama list >/dev/null 2>&1 && break; sleep 1; done
for model in bge-m3 bespoke-minicheck; do
  log "pulling $model"
  ollama pull "$model" >/dev/null || { log "could not pull $model"; exit 1; }
done

log "downloading the passage index"
$SYNC download "$PREFIX/index/passages.sqlite" /work/index/passages.sqlite || exit 1

n=0
failed=0
while IFS= read -r question || [ -n "$question" ]; do
  question="$(printf '%s' "$question" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  case "$question" in ''|'#'*) continue ;; esac
  n=$((n + 1))
  log "question $n: $question"
  if ! research-pipeline ask "$question" >"/work/answer-$n.md" 2>"/work/question-$n.log"; then
    failed=$((failed + 1))
    log "question $n failed: $(tail -1 "/work/question-$n.log")"
  fi
done <"$QUESTIONS"

python - <<'EOF'
import json, pathlib
runs = sorted(pathlib.Path("/work/runs").glob("*/run.json"))
rows = []
for path in runs:
    log = json.loads(path.read_text())
    endpoint = log.get("endpoint") or {}
    kept = sum(c["verdict"] in ("SUPPORTED", "DISPUTED") for c in log["claims"])
    rows.append({
        "question": log["question"],
        "run": path.parent.name,
        "seconds": log["seconds"],
        "claims_kept": kept,
        "claims_drafted": len(log["claims"]),
        "specialist": (log.get("agent_route") or {}).get("specialist"),
        "model_cost_usd": endpoint.get("cost_usd"),
        "providers": endpoint.get("providers"),
        "papers": sorted({e["work"] for e in log["evidence"]}),
    })
pathlib.Path("/work/batch.json").write_text(json.dumps(rows, indent=1))
print(f"{len(rows)} run(s) written")
EOF

log "uploading results"
$SYNC put-dir /work/runs "$PREFIX/batches/$BATCH_ID/runs" || log "results upload failed"
log "done: $n question(s), $failed failed"
[ "$failed" -eq 0 ]
