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
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "PIPELINE_EMBEDDING_MODEL", os.environ.get("EMBEDDING_MODEL", "bge-m3")
        )
    )
    num_ctx: int = field(default_factory=lambda: _int("PIPELINE_NUM_CTX", 16384))
    reranker: str = field(default_factory=lambda: os.environ.get("PIPELINE_RERANKER", "auto"))
    # "<hub repo>::<onnx file>". int8 bge-reranker-v2-m3 is a 571 MB download on first use.
    reranker_model: str = field(
        default_factory=lambda: os.environ.get(
            "PIPELINE_RERANKER_MODEL",
            "onnx-community/bge-reranker-v2-m3-ONNX::onnx/model_int8.onnx",
        )
    )

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

    # Budgets. Every one of these bounds work on a slow local model; none is a quality claim.
    max_subquestions: int = field(default_factory=lambda: _int("PIPELINE_MAX_SUBQUESTIONS", 4))
    queries_per_subquestion: int = field(
        default_factory=lambda: _int("PIPELINE_QUERIES_PER_SUBQ", 2)
    )
    hits_per_query: int = field(default_factory=lambda: _int("PIPELINE_HITS_PER_QUERY", 8))
    max_fetch: int = field(default_factory=lambda: _int("PIPELINE_MAX_FETCH", 12))
    candidates_per_subquestion: int = field(default_factory=lambda: _int("PIPELINE_CANDIDATES", 40))
    passages_per_subquestion: int = field(default_factory=lambda: _int("PIPELINE_PASSAGES", 6))
    max_passages_per_paper: int = field(default_factory=lambda: _int("PIPELINE_MAX_PER_PAPER", 2))
    max_rounds: int = field(default_factory=lambda: _int("PIPELINE_MAX_ROUNDS", 2))
    # discover() and acquire() are network-bound (paper-search providers, PDF fetch/parse); running
    # several at once shortens wall-clock time without adding any extra model or DB work per item.
    search_concurrency: int = field(default_factory=lambda: _int("PIPELINE_SEARCH_CONCURRENCY", 4))
    fetch_concurrency: int = field(default_factory=lambda: _int("PIPELINE_FETCH_CONCURRENCY", 4))

    @property
    def index_path(self) -> Path:
        return self.data_dir / "passages.sqlite"
