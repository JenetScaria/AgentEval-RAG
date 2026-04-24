"""
FastAPI server for AgentEval RAG.

Run:
    uvicorn api:app --reload
"""
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="AgentEval RAG API",
    description="Self-evaluating agentic RAG pipeline powered by Claude",
    version="1.0.0",
)


# ── Lazy graph loading ────────────────────────────────────────────────────────

_run_query = None


def _get_runner():
    global _run_query
    if _run_query is None:
        from graph import run_query

        _run_query = run_query
    return _run_query


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    query_type: str
    eval_scores: Dict[str, float]
    retry_count: int
    used_web: bool


class HealthResponse(BaseModel):
    status: str
    model: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> Dict[str, Any]:
    from config import settings

    return {"status": "ok", "model": settings.claude_model}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> Dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be empty")
    try:
        return _get_runner()(request.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ingest")
def ingest_documents() -> Dict[str, str]:
    """Trigger document ingestion (re-build FAISS + BM25 indexes)."""
    try:
        from src.utils.ingest import ingest

        ingest()
        # Reload the retriever singleton after re-indexing
        import graph as g

        if g._retriever is not None:
            g._retriever.reload()

        return {"status": "ingestion complete"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
