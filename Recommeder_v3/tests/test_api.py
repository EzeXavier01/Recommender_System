from fastapi.testclient import TestClient
from app.main import app


def test_health_after_startup():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_recommend_returns_k_items():
    with TestClient(app) as client:
        r = client.get("/recommend/5?k=5")
        assert r.status_code == 200
        body = r.json()
        assert body["is_cold_start"] is False
        assert len(body["recommendations"]) == 5


def test_recommend_cold_start_user():
    with TestClient(app) as client:
        r = client.get("/recommend/999999?k=3")
        assert r.status_code == 200
        assert r.json()["is_cold_start"] is True


def test_recommend_rejects_invalid_k():
    with TestClient(app) as client:
        assert client.get("/recommend/1?k=0").status_code == 400
        assert client.get("/recommend/1?k=1000").status_code == 400


def test_score_new_item_endpoint():
    with TestClient(app) as client:
        r = client.post("/score_new_item/5", json={"description": "HANDMADE CERAMIC PLANT POT"})
        assert r.status_code == 200
        body = r.json()
        assert "predicted_score" in body
        assert 0 <= body["percentile_vs_existing_catalogue"] <= 100


def test_score_new_item_rejects_unknown_user():
    with TestClient(app) as client:
        r = client.post("/score_new_item/999999", json={"description": "ANYTHING"})
        assert r.status_code == 400
