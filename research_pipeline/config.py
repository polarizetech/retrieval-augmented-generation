"""Settings, read from the environment (and config/rag.env) with conservative defaults.

Defaults are sized for a machine with 16 GB of memory running one ~4B model at a time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / "config/rag.env") -> None:
    """Read KEY=value lines into the environment. Variables already set take precedence."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _path(name: str, default: str) -> Path:
    p = Path(os.environ.get(name, default))
    return p if p.is_absolute() else ROOT / p


def _ollama_url() -> str:
    explicit = os.environ.get("PIPELINE_OLLAMA_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    # OLLAMA_HOST is the server's bind address; 0.0.0.0 is not a destination.
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("0.0.0.0", "127.0.0.1")  # noqa: S104
    return host if host.startswith("http") else f"http://{host}"


@dataclass
class Settings:
    ollama_url: str = field(default_factory=_ollama_url)
    text_model: str = field(
        default_factory=lambda: os.environ.get(
            "PIPELINE_TEXT_MODEL", os.environ.get("TEXT_MODEL", "qwen3:4b-instruct-2507-q4_K_M")
        )
    )
    # The verifier should differ from the generator where possible: a model grading its own
    # output shares its blind spots. Empty means "fall back to the text model and say so".
    verifier_model: str = field(
        default_factory=lambda: os.environ.get("PIPELINE_VERIFIER_MODEL", "")
    )
    num_ctx: int = field(default_factory=lambda: _int("PIPELINE_NUM_CTX", 16384))
    # Most tokens one structured call may generate. The largest legitimate answer (a plan or a
    # synthesis) is a few hundred; without a cap, a call whose model never closes its answer ran
    # for the whole 600 s wall-clock limit (observed: 16,000 tokens, none of them output).
    max_output_tokens: int = field(default_factory=lambda: _int("PIPELINE_MAX_OUTPUT_TOKENS", 2048))
    # How long Ollama keeps a model loaded after a call. Its default (5 minutes) unloads the text
    # model during a run's network-bound stages, and every reload costs seconds.
    ollama_keep_alive: str = field(
        default_factory=lambda: os.environ.get("PIPELINE_OLLAMA_KEEP_ALIVE", "30m")
    )
    # Behind the MCP server: "client" = the calling model fills in the forms; "ollama" = local.
    mcp_llm: str = field(default_factory=lambda: os.environ.get("PIPELINE_MCP_LLM", "client"))
    # Seconds a pipeline tool call waits for the run to need the client again before returning.
    client_turn_wait: int = field(default_factory=lambda: _int("PIPELINE_CLIENT_TURN_WAIT", 40))
    # Most tasks handed to the client in one turn.
    client_batch: int = field(default_factory=lambda: _int("PIPELINE_CLIENT_BATCH", 12))
    # A run whose client stops answering fails after this long instead of waiting forever.
    client_timeout: int = field(default_factory=lambda: _int("PIPELINE_CLIENT_TIMEOUT", 1800))

    data_dir: Path = field(default_factory=lambda: _path("PIPELINE_DATA_DIR", "data/pipeline"))
    runs_dir: Path = field(default_factory=lambda: _path("PIPELINE_RUNS_DIR", "runs"))
    gateway_config: Path = field(
        default_factory=lambda: _path("MCP_CONFIG", "config/mcp-gateway.json")
    )
    papers_upstream: str = field(
        default_factory=lambda: os.environ.get("PIPELINE_PAPERS_UPSTREAM", "papers")
    )

    # The field to research, by slug ("cardiovascular") or distribution name. Empty means the
    # generic policy: the engine answers, without a field's vocabulary or grading rules.
    domain: str = field(default_factory=lambda: os.environ.get("PIPELINE_DOMAIN", ""))
    # A paper-library collection to work in: searches and fetches are recorded there, and
    # retrieval reads only its papers. Empty: retrieval reads every indexed paper.
    collection: str = field(default_factory=lambda: os.environ.get("PIPELINE_COLLECTION", ""))

    # Budgets. Every one of these bounds work on a slow local model; none is a quality claim.
    max_subquestions: int = field(default_factory=lambda: _int("PIPELINE_MAX_SUBQUESTIONS", 4))
    queries_per_subquestion: int = field(
        default_factory=lambda: _int("PIPELINE_QUERIES_PER_SUBQ", 2)
    )
    hits_per_query: int = field(default_factory=lambda: _int("PIPELINE_HITS_PER_QUERY", 8))
    max_fetch: int = field(default_factory=lambda: _int("PIPELINE_MAX_FETCH", 12))
    # Passages asked of the library per retrieval query. The library ranks them (BM25 + vectors,
    # then its cross-encoder when PAPER_FETCH_RERANK_MODEL is set); nothing here re-scores them.
    candidates_per_subquestion: int = field(default_factory=lambda: _int("PIPELINE_CANDIDATES", 12))
    passages_per_subquestion: int = field(default_factory=lambda: _int("PIPELINE_PASSAGES", 6))
    max_passages_per_paper: int = field(default_factory=lambda: _int("PIPELINE_MAX_PER_PAPER", 2))
    max_rounds: int = field(default_factory=lambda: _int("PIPELINE_MAX_ROUNDS", 2))
    # discover() and acquire() are network-bound (paper-search providers, PDF fetch/parse); running
    # several at once shortens wall-clock time without adding any extra model or DB work per item.
    search_concurrency: int = field(default_factory=lambda: _int("PIPELINE_SEARCH_CONCURRENCY", 4))
    fetch_concurrency: int = field(default_factory=lambda: _int("PIPELINE_FETCH_CONCURRENCY", 4))

    @property
    def notes_path(self) -> Path:
        return self.data_dir / "notes.sqlite"
