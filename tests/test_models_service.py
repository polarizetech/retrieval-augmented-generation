"""The model router's choice of service and the model service's lazy reranker (deploy/)."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

DEPLOY = Path(__file__).parent.parent / "deploy"


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, DEPLOY / f"{name}.py")
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_router_tries_local_first_by_default() -> None:
    router = load("models_router")
    providers = router.providers_from_env(
        {"MODELS_LOCAL_URL": "https://mac/", "MODELS_MODAL_URL": "https://modal"}
    )
    assert [p.name for p in providers] == ["local", "modal"]
    assert providers[0].url == "https://mac"


def test_a_provider_that_is_out_sits_out_for_the_cooldown() -> None:
    router = load("models_router")
    now = [0.0]
    a, b, c = (router.Provider(name, f"https://{name}", "") for name in "abc")
    ps = router.Providers([a, b, c], 900.0, lambda: now[0])
    assert [p.name for p in ps.order()] == ["a", "b", "c"]
    ps.mark_out(b)
    assert [p.name for p in ps.order()] == ["a", "c", "b"]
    assert ps.status()["out_for_seconds"] == {"b": 900.0}
    now[0] = 901.0
    assert [p.name for p in ps.order()] == ["a", "b", "c"]


def test_all_providers_out_are_still_tried() -> None:
    router = load("models_router")
    now = [0.0]
    a, b = (router.Provider(name, f"https://{name}", "") for name in "ab")
    ps = router.Providers([a, b], 900.0, lambda: now[0])
    ps.mark_out(a)
    ps.mark_out(b)
    assert [p.name for p in ps.order()] == ["a", "b"]
    assert ps.status()["using"] == "a"


def test_legacy_settings_map_to_modal_then_local() -> None:
    router = load("models_router")
    providers = router.providers_from_env(
        {
            "MODELS_PRIMARY_URL": "https://modal",
            "MODELS_FALLBACK_URL": "https://mac",
            "MODELS_FALLBACK_PROXY": "http://127.0.0.1:1055",
        }
    )
    assert [p.name for p in providers] == ["modal", "local"]
    assert providers[1].proxy == "http://127.0.0.1:1055"


def test_a_missing_provider_url_is_named() -> None:
    router = load("models_router")
    with pytest.raises(KeyError, match="MODELS_DIGITALOCEAN_URL"):
        router.providers_from_env(
            {"MODELS_PROVIDERS": "local,digitalocean", "MODELS_LOCAL_URL": "https://mac"}
        )


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
