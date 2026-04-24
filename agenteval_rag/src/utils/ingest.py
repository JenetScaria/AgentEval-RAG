"""
Ingestion pipeline: Amazon QA dataset → chunks → FAISS + BM25 indexes.

Downloads McAuley-Lab/Amazon-C4 from HuggingFace, chunks the texts, and
builds the FAISS and BM25 indexes that HybridRetriever loads at query time.

Usage:
    python src/utils/ingest.py                   # default 5 000 samples
    python src/utils/ingest.py --samples 10000   # larger run
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[3]))
from config import settings


# ── Dataset download ──────────────────────────────────────────────────────────

def download_amazon_c4(max_samples: int = 5_000) -> list[Document]:
    """
    Stream McAuley-Lab/Amazon-C4 from HuggingFace and return LangChain Documents.

    Amazon-C4 is a cleaned web corpus focused on Amazon product content — good
    for e-commerce / product-knowledge RAG.
    """
    print(f"Downloading Amazon-C4 (up to {max_samples:,} samples, streaming) …")
    dataset = load_dataset(
        "McAuley-Lab/Amazon-C4",
        split="test",
        streaming=True,
    )

    documents: list[Document] = []
    for i, item in enumerate(tqdm(dataset, total=max_samples, desc="Fetching")):
        if i >= max_samples:
            break
        review = (item.get("ori_review") or "").strip()
        query  = (item.get("query") or "").strip()
        if not review:
            continue
        # Combine query + review so the chunk carries both the question context
        # and the answer text, making retrieval more informative.
        text = f"Q: {query}\nReview: {review}" if query else review
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "amazon_c4",
                    "idx": i,
                    "item_id": item.get("item_id", ""),
                    "rating": item.get("ori_rating", ""),
                },
            )
        )

    print(f"  Fetched {len(documents):,} documents.")
    return documents


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks):,} chunks from {len(documents):,} documents.")
    return chunks


# ── Index building ────────────────────────────────────────────────────────────

def build_faiss_index(chunks: list[Document]) -> faiss.Index:
    """
    Encode chunks with SentenceTransformer and build a FAISS flat IP index.

    Uses the same model + normalization as HybridRetriever so query vectors
    are comparable at retrieval time.
    """
    model = SentenceTransformer(settings.embedding_model)
    texts = [c.page_content for c in chunks]

    print(f"  Encoding {len(texts):,} chunks with '{settings.embedding_model}' …")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # cosine via inner product
        batch_size=64,
        show_progress_bar=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"  FAISS index: {index.ntotal:,} vectors, dim={embeddings.shape[1]}")
    return index


def build_bm25_index(chunks: list[Document]) -> BM25Okapi:
    """
    Build a BM25 index with regex tokenization (handles punctuation correctly).
    """
    tokenized = [
        re.findall(r"\b\w+\b", c.page_content.lower())
        for c in chunks
    ]
    return BM25Okapi(tokenized)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_indexes(
    faiss_index: faiss.Index,
    bm25_index: BM25Okapi,
    chunks: list[Document],
) -> None:
    processed_dir = settings.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = settings.faiss_index_path.with_suffix(".faiss")
    faiss.write_index(faiss_index, str(faiss_path))

    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump(bm25_index, f)

    docs_path = Path(str(settings.faiss_index_path) + "_docs.pkl")
    with open(docs_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"  Saved FAISS index  → {faiss_path}")
    print(f"  Saved BM25 index   → {settings.bm25_index_path}")
    print(f"  Saved {len(chunks):,} chunks  → {docs_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def ingest(max_samples: int = 5_000) -> None:
    """Full pipeline: download → chunk → FAISS → BM25 → save."""
    print("=== AgentEval RAG — Ingestion Pipeline ===\n")

    print("[1/4] Downloading Amazon-C4 dataset …")
    docs = download_amazon_c4(max_samples)
    if not docs:
        print("No documents fetched. Check your internet connection and try again.")
        return

    print("\n[2/4] Chunking …")
    chunks = chunk_documents(docs)

    print("\n[3/4] Building FAISS index …")
    faiss_index = build_faiss_index(chunks)

    print("\n[4/4] Building BM25 index and saving …")
    bm25_index = build_bm25_index(chunks)
    save_indexes(faiss_index, bm25_index, chunks)

    print("\n✅ Ingestion complete. Run `streamlit run app.py` to start the app.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentEval RAG ingestion pipeline")
    parser.add_argument(
        "--samples",
        type=int,
        default=5_000,
        help="Number of Amazon-C4 samples to ingest (default: 5000)",
    )
    args = parser.parse_args()
    ingest(max_samples=args.samples)
