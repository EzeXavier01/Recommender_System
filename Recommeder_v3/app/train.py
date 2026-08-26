"""
Trains the production model — implicit ALS + content similarity, blended at alpha=0.7
(tuned in the notebook) — on the real Online Retail dataset, and saves the artifacts the
API needs to serve recommendations without retraining on every request.

Run with: python -m app.train
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from implicit.als import AlternatingLeastSquares

from app.download_data import download, DATA_PATH

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "model_artifacts")


def train_and_save():
    download()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df[df.UnitPrice > 0].copy()
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    agg = df.groupby(["CustomerID", "StockCode"]).agg(n_purchases=("InvoiceNo", "nunique")).reset_index()
    agg["strength"] = 1 + np.log1p(agg["n_purchases"])

    user_ids = sorted(agg.CustomerID.unique())
    item_ids = sorted(agg.StockCode.unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {it: i for i, it in enumerate(item_ids)}
    n_users, n_items = len(user_ids), len(item_ids)

    agg["u_idx"] = agg.CustomerID.map(user_map)
    agg["i_idx"] = agg.StockCode.map(item_map)

    train_matrix = csr_matrix((agg.strength, (agg.u_idx, agg.i_idx)), shape=(n_users, n_items))
    print(f"Training on {train_matrix.nnz:,} interactions ({n_users} users x {n_items} items)")

    als_model = AlternatingLeastSquares(factors=20, regularization=0.04, iterations=20, random_state=42)
    als_model.fit(train_matrix)

    desc_by_item = (
        df[["StockCode", "Description"]]
        .drop_duplicates("StockCode")
        .assign(i_idx=lambda d: d["StockCode"].map(item_map))
        .dropna(subset=["i_idx"])
        .set_index("i_idx")["Description"]
        .reindex(range(n_items))
        .fillna("")
    )
    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    text_features = tfidf.fit_transform(desc_by_item.values)
    content_sim = cosine_similarity(text_features)
    np.fill_diagonal(content_sim, 0)

    item_pop = np.asarray(train_matrix.sum(axis=0)).flatten()
    popularity_ranking = np.argsort(-item_pop)

    with open(os.path.join(ARTIFACT_DIR, "als_model.pkl"), "wb") as f:
        pickle.dump(als_model, f)
    with open(os.path.join(ARTIFACT_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(ARTIFACT_DIR, "text_features.pkl"), "wb") as f:
        pickle.dump(text_features, f)
    np.save(os.path.join(ARTIFACT_DIR, "content_sim.npy"), content_sim)
    save_npz(os.path.join(ARTIFACT_DIR, "train_matrix.npz"), train_matrix)
    np.save(os.path.join(ARTIFACT_DIR, "popularity_ranking.npy"), popularity_ranking)
    np.save(os.path.join(ARTIFACT_DIR, "meta.npy"), np.array([n_users, n_items]))

    item_lookup = df[["StockCode", "Description"]].drop_duplicates("StockCode")
    item_lookup["i_idx"] = item_lookup["StockCode"].map(item_map)
    item_lookup = item_lookup.dropna(subset=["i_idx"]).set_index("i_idx")["Description"]
    item_lookup = item_lookup.reindex(range(n_items)).fillna("Unknown item")
    with open(os.path.join(ARTIFACT_DIR, "item_lookup.pkl"), "wb") as f:
        pickle.dump(item_lookup.to_dict(), f)

    print(f"Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    train_and_save()
