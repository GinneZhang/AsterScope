import os

from fastapi.testclient import TestClient

from retrieval.main import app


class _StubRetriever:
    def __init__(self):
        self.last_search_debug = {"candidate_pool_count": 2, "final_result_count": 1}
        self.enable_graph_retrieval = True
        self.pg_conn = type("Conn", (), {"closed": 0})()
        self.neo4j_driver = object()

    def collect_candidate_pool(self, query, top_k=5, additional_queries=None, include_follow_ups=True):
        return [
            {
                "doc_id": "doc-1",
                "chunk_index": 0,
                "title": "Test Doc",
                "chunk_text": "retrieved context",
                "score": 0.9,
                "source": "dense",
            }
        ]

    def finalize_candidates(self, query, candidate_pool, top_k=5, query_graph=None):
        return candidate_pool[:1]


def test_retrieval_health(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr("retrieval.main._get_retriever", lambda: _StubRetriever())
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "AsterScope Retrieval API"
    assert payload["postgres_connected"] is True


def test_retrieval_search(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setattr("retrieval.main._get_retriever", lambda: _StubRetriever())
    client = TestClient(app)

    response = client.post(
        "/search",
        headers={"X-API-KEY": "test-key"},
        json={"query": "test query", "top_k": 3},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "test query"
    assert payload["top_k"] == 3
    assert len(payload["results"]) == 1
    assert payload["results"][0]["title"] == "Test Doc"
    assert payload["debug_metrics"]["candidate_pool_count"] == 2
