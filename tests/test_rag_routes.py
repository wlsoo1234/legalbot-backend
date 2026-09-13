from app.main import app


def test_canonical_routes_are_mounted_and_internal_routes_are_not():
    paths = {route.path for route in app.routes}
    assert "/api/v1/chat/ask" in paths
    assert "/api/v1/assistant/consultation/send" in paths
    assert "/api/v1/rag/sources" in paths
    assert "/api/v1/rag/healthz" in paths
    assert "/api/v1/rag/readyz" in paths
    assert "/api/v1/rag/ask" not in paths
    assert not any(path == "/d4" or path.startswith("/d4/") for path in paths)
    assert "/api/v1/consultation" not in paths
