"""
Retrieval tests — require built indexes (run src/utils/ingest.py first).

Run:
    pytest tests/test_retrieval.py -v
"""
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_DOCS = [
    Document(
        page_content="Transformers use self-attention to model long-range dependencies.",
        metadata={"source": "paper.pdf", "page": 1},
    ),
    Document(
        page_content="BERT is a bidirectional encoder representation from transformers.",
        metadata={"source": "paper.pdf", "page": 2},
    ),
    Document(
        page_content="GPT uses a unidirectional transformer decoder architecture.",
        metadata={"source": "gpt.pdf", "page": 1},
    ),
    Document(
        page_content="Retrieval-augmented generation improves factual accuracy.",
        metadata={"source": "rag.pdf", "page": 1},
    ),
    Document(
        page_content="BM25 is a classical sparse retrieval algorithm based on TF-IDF.",
        metadata={"source": "ir.pdf", "page": 5},
    ),
]


@pytest.fixture()
def mock_retriever():
    """HybridRetriever with in-memory indexes (no disk required)."""
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    from src.retrieval.hybrid_retriever import HybridRetriever

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [d.page_content for d in SAMPLE_DOCS]
    embeddings = model.encode(texts, normalize_embeddings=True).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    bm25 = BM25Okapi([t.lower().split() for t in texts])

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.embedding_model = model
    from sentence_transformers import CrossEncoder

    retriever.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    retriever.faiss_index = index
    retriever.bm25_index = bm25
    retriever.documents = SAMPLE_DOCS
    return retriever


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHybridRetriever:
    def test_returns_documents(self, mock_retriever):
        results = mock_retriever.retrieve("transformer attention mechanism", top_k=3)
        assert len(results) > 0
        assert all(isinstance(d, Document) for d in results)

    def test_top_k_respected(self, mock_retriever):
        results = mock_retriever.retrieve("BERT GPT language model", top_k=2)
        assert len(results) <= 2

    def test_relevant_doc_in_top_results(self, mock_retriever):
        results = mock_retriever.retrieve("BM25 sparse retrieval TF-IDF", top_k=3)
        contents = [r.page_content for r in results]
        assert any("BM25" in c or "sparse" in c.lower() for c in contents)

    def test_empty_index_returns_empty(self):
        from src.retrieval.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.faiss_index = None
        retriever.bm25_index = None
        retriever.documents = []
        from sentence_transformers import SentenceTransformer, CrossEncoder

        retriever.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        retriever.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        results = retriever.retrieve("anything")
        assert results == []

    def test_faiss_search_scores_normalised(self, mock_retriever):
        results = mock_retriever._faiss_search("attention", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_bm25_search_scores_normalised(self, mock_retriever):
        results = mock_retriever._bm25_search("transformer", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0


class TestRAGASEvaluator:
    def test_heuristic_scores_in_range(self):
        from src.evaluation.ragas_eval import RAGASEvaluator

        ev = RAGASEvaluator()
        scores = ev._heuristic_scores(
            question="What is BM25?",
            answer="BM25 is a ranking function based on TF-IDF.",
            contexts=["BM25 is a sparse retrieval algorithm based on TF-IDF scoring."],
        )
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_passes_threshold_true(self):
        from src.evaluation.ragas_eval import RAGASEvaluator

        ev = RAGASEvaluator()
        good = {"faithfulness": 0.9, "context_precision": 0.85, "answer_relevancy": 0.8}
        assert ev.passes_threshold(good) is True

    def test_passes_threshold_false(self):
        from src.evaluation.ragas_eval import RAGASEvaluator

        ev = RAGASEvaluator()
        poor = {"faithfulness": 0.3, "context_precision": 0.9, "answer_relevancy": 0.9}
        assert ev.passes_threshold(poor) is False

    def test_evaluate_returns_overall(self):
        from src.evaluation.ragas_eval import RAGASEvaluator

        ev = RAGASEvaluator()
        scores = ev.evaluate(
            question="What is RAG?",
            answer="RAG combines retrieval with generation.",
            contexts=["Retrieval-augmented generation (RAG) improves factual accuracy."],
        )
        assert "overall" in scores
        assert 0.0 <= scores["overall"] <= 1.0
