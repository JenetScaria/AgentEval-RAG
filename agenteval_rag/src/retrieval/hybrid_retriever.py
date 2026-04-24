"""
Hybrid retriever: FAISS (dense) + BM25 (sparse) + cross-encoder reranking.

Usage:
    retriever = HybridRetriever()
    docs = retriever.retrieve("What is transformer architecture?")
"""
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import settings


class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.cross_encoder = CrossEncoder(settings.reranker_model)
        self.faiss_index: faiss.Index | None = None
        self.bm25_index: BM25Okapi | None = None
        self.documents: List[Document] = []
        self._load_indexes()

    # ── Index loading ─────────────────────────────────────────────────────────

    def _load_indexes(self) -> None:
        faiss_path = settings.faiss_index_path.with_suffix(".faiss")
        bm25_path = settings.bm25_index_path
        docs_path = Path(str(settings.faiss_index_path) + "_docs.pkl")

        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))

        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                self.bm25_index = pickle.load(f)

        if docs_path.exists():
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)

    def reload(self) -> None:
        """Reload indexes from disk (call after running ingest.py)."""
        self._load_indexes()

    # ── Public retrieval ──────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int | None = None) -> List[Document]:
        """
        Hybrid retrieval pipeline:
          1. FAISS dense search (60 % weight)
          2. BM25 sparse search (40 % weight)
          3. Reciprocal-rank fusion merge
          4. Cross-encoder reranking of top-k candidates
        """
        if top_k is None:
            top_k = settings.top_k_retrieval

        if not self.documents:
            return []

        dense = self._faiss_search(query, top_k)
        sparse = self._bm25_search(query, top_k)

        # Weighted score fusion
        fused: Dict[int, float] = {}
        for idx, score in dense:
            fused[idx] = fused.get(idx, 0.0) + score * 0.6
        for idx, score in sparse:
            fused[idx] = fused.get(idx, 0.0) + score * 0.4

        top_indices = sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]
        candidates = [
            self.documents[i] for i in top_indices if i < len(self.documents)
        ]

        return self._rerank(query, candidates)

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _faiss_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.faiss_index is None:
            return []
        embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        k = min(top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(embedding, k)
        return [
            (int(idx), float(1.0 / (1.0 + dist)))
            for dist, idx in zip(distances[0], indices[0])
            if idx != -1
        ]

    # ── Sparse retrieval ──────────────────────────────────────────────────────

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.bm25_index is None:
            return []
        scores: np.ndarray = self.bm25_index.get_scores(query.lower().split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
        if results:
            max_score = max(s for _, s in results)
            if max_score > 0:
                results = [(i, s / max_score) for i, s in results]
        return results

    # ── Cross-encoder reranking ───────────────────────────────────────────────

    def _rerank(self, query: str, candidates: List[Document]) -> List[Document]:
        top_k_rerank = settings.top_k_rerank
        if len(candidates) <= top_k_rerank:
            return candidates
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:top_k_rerank]]
