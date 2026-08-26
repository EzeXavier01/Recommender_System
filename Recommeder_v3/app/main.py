"""
Serving layer for the marketplace recommender (implicit ALS + content, weighted hybrid).

Run locally with: uvicorn app.main:app --reload
"""
import os
import pickle
from contextlib import asynccontextmanager
import numpy as np
from scipy.sparse import load_npz
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.recommender import recommend_for_user, score_new_item_for_user, ALPHA

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")

_state = {}


def load_model():
    meta = np.load(os.path.join(ARTIFACT_DIR, "meta.npy"))
    n_users, n_items = int(meta[0]), int(meta[1])

    with open(os.path.join(ARTIFACT_DIR, "als_model.pkl"), "rb") as f:
        als_model = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "text_features.pkl"), "rb") as f:
        text_features = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, "item_lookup.pkl"), "rb") as f:
        item_lookup = pickle.load(f)

    _state["als_model"] = als_model
    _state["tfidf"] = tfidf
    _state["text_features"] = text_features
    _state["content_sim"] = np.load(os.path.join(ARTIFACT_DIR, "content_sim.npy"))
    _state["train_matrix"] = load_npz(os.path.join(ARTIFACT_DIR, "train_matrix.npz"))
    _state["popularity_ranking"] = np.load(os.path.join(ARTIFACT_DIR, "popularity_ranking.npy"))
    _state["item_lookup"] = item_lookup
    _state["n_users"] = n_users
    _state["n_items"] = n_items


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    _state.clear()


app = FastAPI(
    title="Marketplace Recommender API (Weighted Hybrid: implicit ALS + Content)",
    description="Serves top-K item recommendations using a tuned hybrid of implicit-feedback "
                "ALS and TF-IDF content similarity (alpha=0.7), trained on real transaction data.",
    version="2.0.0",
    lifespan=lifespan,
)


class RecommendedItem(BaseModel):
    item_index: int
    description: str


class RecommendationResponse(BaseModel):
    user_id: int
    is_cold_start: bool
    recommendations: list[RecommendedItem]


@app.get("/health")
def health():
    return {"status": "ok" if "als_model" in _state else "loading"}


@app.get("/recommend/{user_idx}", response_model=RecommendationResponse)
def recommend(user_idx: int, k: int = 10):
    if "als_model" not in _state:
        raise HTTPException(status_code=503, detail="Model not yet loaded")
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")

    is_cold_start = user_idx < 0 or user_idx >= _state["n_users"]
    rec_ids = recommend_for_user(
        user_idx=user_idx,
        n_users=_state["n_users"],
        als_model=_state["als_model"],
        content_sim=_state["content_sim"],
        train_matrix=_state["train_matrix"],
        popularity_ranking=_state["popularity_ranking"],
        k=k,
        alpha=ALPHA,
    )
    items = [
        RecommendedItem(item_index=i, description=_state["item_lookup"].get(i, "Unknown item"))
        for i in rec_ids
    ]
    return RecommendationResponse(user_id=user_idx, is_cold_start=is_cold_start, recommendations=items)


class NewItemRequest(BaseModel):
    description: str


class NewItemScoreResponse(BaseModel):
    user_id: int
    description: str
    predicted_score: float
    percentile_vs_existing_catalogue: float


@app.post("/score_new_item/{user_idx}", response_model=NewItemScoreResponse)
def score_new_item(user_idx: int, request: NewItemRequest):
    """Cold-start endpoint: score a brand-new item (zero purchase history) for a user,
    using only its description text — see recommender.score_new_item_for_user()."""
    if "als_model" not in _state or user_idx < 0 or user_idx >= _state["n_users"]:
        raise HTTPException(status_code=400, detail="Unknown user_idx; this endpoint requires a known user")

    score, _, _ = score_new_item_for_user(
        user_idx=user_idx,
        new_item_text=request.description,
        tfidf=_state["tfidf"],
        text_features=_state["text_features"],
        als_model=_state["als_model"],
    )
    existing_scores = _state["als_model"].item_factors @ _state["als_model"].user_factors[user_idx]
    percentile = float((existing_scores < score).mean() * 100)
    return NewItemScoreResponse(
        user_id=user_idx, description=request.description,
        predicted_score=float(score), percentile_vs_existing_catalogue=percentile,
    )
