"""The model router's choice of service and the model service's lazy reranker (deploy/)."""

import importlib.util
import json
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


def test_digitalocean_maps_rerank_and_embeddings() -> None:
    router = load("models_router")
    p = router.Provider("digitalocean", "https://do", "", kind="digitalocean", key="do-key")
    sent = router.digitalocean_request(
        "POST",
        "/rerank",
        b'{"model": "onnx-community/bge-reranker-v2-m3-ONNX::onnx/model.onnx", '
        b'"query": "q", "documents": ["a", "b"]}',
        p,
    )
    assert sent == (
        "v1/rerank",
        {"model": "bge-reranker-v2-m3", "query": "q", "documents": ["a", "b"]},
        "rerank",
    )
    openai = router.digitalocean_request("POST", "v1/embeddings", b'{"input": ["x"]}', p)
    assert openai == ("v1/embeddings", {"model": "bge-m3", "input": ["x"]}, "openai")
    ollama = router.digitalocean_request("POST", "api/embed", b'{"input": ["x"]}', p)
    assert ollama == ("v1/embeddings", {"model": "bge-m3", "input": ["x"]}, "ollama")
    one = router.digitalocean_request("POST", "api/embed", b'{"input": "x"}', p)
    assert one is not None
    assert one[1]["input"] == ["x"]


def test_digitalocean_does_not_serve_chat_or_tags() -> None:
    router = load("models_router")
    p = router.Provider("digitalocean", "https://do", "", kind="digitalocean")
    assert router.digitalocean_request("POST", "api/chat", b"{}", p) is None
    assert router.digitalocean_request("GET", "api/tags", b"", p) is None
    assert router.digitalocean_request("GET", "api/ps", b"", p) is None


def test_digitalocean_rerank_scores_come_back_in_document_order() -> None:
    router = load("models_router")
    body = {
        "results": [{"index": 1, "relevance_score": -2.0}, {"index": 0, "relevance_score": 3.5}]
    }
    assert router.digitalocean_response("rerank", body, "bge-m3") == {"scores": [3.5, -2.0]}


def test_digitalocean_embeddings_are_unit_length() -> None:
    router = load("models_router")
    body = {"object": "list", "model": "bge-m3", "data": [{"index": 0, "embedding": [3.0, 4.0]}]}
    openai = router.digitalocean_response("openai", body, "bge-m3")
    assert openai["data"][0]["embedding"] == pytest.approx([0.6, 0.8], abs=1e-9)
    ollama = router.digitalocean_response("ollama", body, "bge-m3")
    assert set(ollama) == {"model", "embeddings"}
    assert ollama["model"] == "bge-m3"
    assert ollama["embeddings"][0] == pytest.approx([0.6, 0.8], abs=1e-9)


def test_router_sends_rerank_to_digitalocean_with_its_own_key() -> None:
    import httpx
    from fastapi.testclient import TestClient

    router = load("models_router")
    seen: dict[str, list] = {"do": [], "local": []}

    def do_handler(request: httpx.Request) -> httpx.Response:
        seen["do"].append((request.headers.get("authorization"), request.url.path))
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": -2.0},
                    {"index": 0, "relevance_score": 3.5},
                ]
            },
        )

    def local_handler(request: httpx.Request) -> httpx.Response:
        seen["local"].append(request.url.path)
        return httpx.Response(200, json={"scores": [0.0]})

    providers = [
        router.Provider("digitalocean", "https://do", "", kind="digitalocean", key="do-key"),
        router.Provider("local", "https://mac", ""),
    ]
    transports = {
        "digitalocean": httpx.MockTransport(do_handler),
        "local": httpx.MockTransport(local_handler),
    }
    app = router.create_app(router.Providers(providers, 900.0), transports=transports)
    client = TestClient(app)
    reply = client.post(
        "/rerank",
        json={"model": "x", "query": "q", "documents": ["a", "b"]},
        headers={"Authorization": "Bearer caller-key"},
    )
    assert seen["do"] == [("Bearer do-key", "/v1/rerank")]
    assert seen["local"] == []
    assert reply.json() == {"scores": [3.5, -2.0]}


def test_router_passes_chat_past_digitalocean() -> None:
    import httpx
    from fastapi.testclient import TestClient

    router = load("models_router")
    seen: dict[str, list] = {"do": [], "local": []}

    def do_handler(request: httpx.Request) -> httpx.Response:
        seen["do"].append(request.url.path)
        return httpx.Response(200, json={})

    def local_handler(request: httpx.Request) -> httpx.Response:
        seen["local"].append(request.headers.get("authorization"))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({"message": "ok"}).encode()),
        )

    providers = [
        router.Provider("digitalocean", "https://do", "", kind="digitalocean", key="do-key"),
        router.Provider("local", "https://mac", ""),
    ]
    transports = {
        "digitalocean": httpx.MockTransport(do_handler),
        "local": httpx.MockTransport(local_handler),
    }
    ps = router.Providers(providers, 900.0)
    client = TestClient(router.create_app(ps, transports=transports))
    reply = client.post(
        "/api/chat", json={"model": "x"}, headers={"Authorization": "Bearer caller-key"}
    )
    assert reply.json() == {"message": "ok"}
    assert seen["local"] == ["Bearer caller-key"]
    assert seen["do"] == []
    assert "digitalocean" not in ps.status()["out_for_seconds"]


def test_router_falls_back_when_digitalocean_answers_402() -> None:
    import httpx
    from fastapi.testclient import TestClient

    router = load("models_router")

    def do_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": "no credit"})

    def local_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps({"scores": [1.0]}).encode()),
        )

    providers = [
        router.Provider("digitalocean", "https://do", "", kind="digitalocean", key="do-key"),
        router.Provider("local", "https://mac", ""),
    ]
    transports = {
        "digitalocean": httpx.MockTransport(do_handler),
        "local": httpx.MockTransport(local_handler),
    }
    ps = router.Providers(providers, 900.0)
    client = TestClient(router.create_app(ps, transports=transports))
    reply = client.post("/rerank", json={"model": "x", "query": "q", "documents": ["a"]})
    assert reply.json() == {"scores": [1.0]}
    assert "digitalocean" in ps.status()["out_for_seconds"]


def test_kind_and_key_file_come_from_the_environment(tmp_path: Path) -> None:
    router = load("models_router")
    key_file = tmp_path / "do.key"
    key_file.write_text(" abc \n")
    providers = router.providers_from_env(
        {
            "MODELS_PROVIDERS": "digitalocean",
            "MODELS_DIGITALOCEAN_URL": "https://do",
            "MODELS_DIGITALOCEAN_KIND": "digitalocean",
            "MODELS_DIGITALOCEAN_KEY_FILE": str(key_file),
        }
    )
    assert providers[0].kind == "digitalocean"
    assert providers[0].key == "abc"
    with pytest.raises(ValueError, match="MODELS_DIGITALOCEAN_KIND"):
        router.providers_from_env(
            {
                "MODELS_PROVIDERS": "digitalocean",
                "MODELS_DIGITALOCEAN_URL": "https://do",
                "MODELS_DIGITALOCEAN_KIND": "other",
            }
        )


def test_a_route_header_sends_the_call_to_that_provider_only() -> None:
    import httpx
    from fastapi.testclient import TestClient

    router = load("models_router")
    seen = {"local": 0, "modal": 0}

    def handler(name: str, status: int) -> Any:
        def handle(request: httpx.Request) -> httpx.Response:
            seen[name] += 1
            return httpx.Response(
                status,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(json.dumps({"scores": [1.0]}).encode()),
            )

        return handle

    providers = [
        router.Provider("local", "https://mac", ""),
        router.Provider("modal", "https://modal", ""),
    ]
    transports = {
        "local": httpx.MockTransport(handler("local", 200)),
        "modal": httpx.MockTransport(handler("modal", 429)),
    }
    ps = router.Providers(providers, 900.0)
    assert ps.named("mac") is providers[0]
    assert ps.named("gpu") is None
    client = TestClient(router.create_app(ps, transports=transports))
    body = {"model": "x", "query": "q", "documents": ["a"]}
    mac = client.post("/rerank", json=body, headers={"X-Models-Route": "mac"})
    assert mac.json() == {"scores": [1.0]}
    assert seen == {"local": 1, "modal": 0}
    modal = client.post("/rerank", json=body, headers={"X-Models-Route": "modal"})
    assert modal.status_code == 503
    assert seen == {"local": 1, "modal": 1}  # no fallback to the Mac
    unknown = client.post("/rerank", json=body, headers={"X-Models-Route": "gpu"})
    assert unknown.status_code == 400
    plain = client.post("/rerank", json=body)
    assert plain.json() == {"scores": [1.0]}
    assert seen == {"local": 2, "modal": 1}
