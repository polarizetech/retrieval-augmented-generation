"""The model router's choice of service and the model service's lazy reranker (deploy/)."""

import importlib.util
from pathlib import Path
from typing import Any

DEPLOY = Path(__file__).parent.parent / "deploy"


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, DEPLOY / f"{name}.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_uses_the_fallback_while_the_primary_is_out() -> None:
    router = load("models_router")
    now = [0.0]
    up = router.Upstreams("https://modal/", "https://mac", 900.0, lambda: now[0])
    assert up.order() == ["https://modal", "https://mac"]
    up.mark_out("https://mac")
    assert up.order() == ["https://modal", "https://mac"]
    up.mark_out("https://modal")
    assert up.order() == ["https://mac"]
    assert up.status() == {"using": "https://mac", "primary_out_for_seconds": 900.0}
    now[0] = 900.0
    assert up.order() == ["https://modal", "https://mac"]
    assert up.status() == {"using": "https://modal", "primary_out_for_seconds": 0.0}


def test_the_reranker_loads_on_use_and_drops_when_idle() -> None:
    service = load("models_service")
    now = [0.0]
    loads: list[tuple[str, str]] = []

    def loader(repo: str, onnx_file: str) -> object:
        loads.append((repo, onnx_file))
        return object()

    lazy = service.LazyReranker("repo", "file", 600.0, lambda: now[0], loader)
    model = lazy()
    assert lazy() is model
    assert loads == [("repo", "file")]
    now[0] = 599.0
    assert not lazy.drop_if_idle()
    now[0] = 1199.0
    assert lazy.drop_if_idle()
    lazy()
    assert len(loads) == 2
