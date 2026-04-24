"""
LangGraph agent graph — the core orchestration layer.

Pipeline:
    router → retrieval → generation → evaluation
                ↑                          |
                └────── reretrieval ←──────┘  (if score < threshold)

Entry point:
    from graph import run_query
    result = run_query("What is attention mechanism?")
"""
import re
from typing import Any, Dict, List, Literal, Optional

from google import genai
from google.genai import types as genai_types
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from config import settings
from src.evaluation.ragas_eval import RAGASEvaluator
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.mlflow_logger import MLflowLogger


# ── Gemini client setup ───────────────────────────────────────────────────────

_gemini_client = genai.Client(api_key=settings.gemini_api_key)

def _gemini(prompt: str, system: str = "", max_tokens: int = 1024) -> str:
    """Single helper that calls Gemini and returns the response text."""
    response = _gemini_client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system if system else None,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text.strip()


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    query_type: Literal["simple", "multi_hop", "web"]
    retrieved_docs: List[Document]
    answer: str
    citations: List[str]
    eval_scores: Dict[str, float]
    retry_count: int
    use_web: bool
    error: Optional[str]


# ── Singletons ────────────────────────────────────────────────────────────────

_retriever: HybridRetriever | None = None
_evaluator = RAGASEvaluator()
_logger = MLflowLogger()


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# ── Node: Router ──────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    """
    Classify the query as one of:
      - simple    : single-document factual lookup
      - multi_hop : requires reasoning across multiple documents
      - web       : requires current / real-time information
    """
    system = (
        "You are a query routing assistant. Classify the query into exactly one of:\n"
        "  simple    — straightforward fact answerable from a document collection\n"
        "  multi_hop — complex question requiring reasoning across multiple sources\n"
        "  web       — requires recent or real-time information\n"
        "Reply with one word only: simple, multi_hop, or web."
    )
    raw = _gemini(state["query"], system=system, max_tokens=8).lower()
    if "multi" in raw or "hop" in raw:
        query_type: Literal["simple", "multi_hop", "web"] = "multi_hop"
    elif "web" in raw:
        query_type = "web"
    else:
        query_type = "simple"
    return {**state, "query_type": query_type}


# ── Node: Retrieval ───────────────────────────────────────────────────────────

def retrieval_node(state: AgentState) -> AgentState:
    """
    Dispatch to the right retrieval strategy:
      - web       → web search (Tavily → DuckDuckGo fallback)
      - multi_hop → iterative two-hop retrieval
      - simple    → single-pass hybrid retrieval
    """
    retriever = _get_retriever()
    query = state["query"]

    if state.get("use_web") or state["query_type"] == "web":
        docs = _web_search(query)
    elif state["query_type"] == "multi_hop":
        docs = _multi_hop_retrieve(query, retriever)
    else:
        docs = retriever.retrieve(query)

    return {**state, "retrieved_docs": docs}


def _web_search(query: str) -> List[Document]:
    """Try Tavily, then DuckDuckGo as fallback."""
    # Tavily
    try:
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=settings.tavily_api_key)
        results = client.search(query, max_results=5).get("results", [])
        return [
            Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", ""), "title": r.get("title", "")},
            )
            for r in results
        ]
    except Exception:
        pass

    # DuckDuckGo
    try:
        from duckduckgo_search import DDGS  # type: ignore

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return [
            Document(
                page_content=r.get("body", ""),
                metadata={"source": r.get("href", ""), "title": r.get("title", "")},
            )
            for r in results
        ]
    except Exception:
        return []


def _multi_hop_retrieve(query: str, retriever: HybridRetriever) -> List[Document]:
    """
    Two-hop iterative retrieval:
      Hop 1 → retrieve initial docs
      Hop 2 → ask Gemini for a follow-up query, retrieve again
    """
    seen: Dict[str, Document] = {}
    current_query = query

    for hop in range(2):
        batch = retriever.retrieve(current_query, top_k=settings.top_k_retrieval // 2)
        for doc in batch:
            key = doc.page_content[:120]
            seen.setdefault(key, doc)

        if hop == 0 and seen:
            context_snippet = "\n".join(
                d.page_content[:200] for d in list(seen.values())[:3]
            )
            current_query = _gemini(
                prompt=(
                    f"Original question: {query}\n\n"
                    f"Context so far:\n{context_snippet}\n\n"
                    "Write a single follow-up search query to fill information gaps. "
                    "Return only the query text."
                ),
                max_tokens=64,
            )

    return list(seen.values())[: settings.top_k_rerank]


# ── Node: Generation ──────────────────────────────────────────────────────────

def generation_node(state: AgentState) -> AgentState:
    """
    Generate a grounded answer using Gemini.
    Citations are tracked as [N] inline references.
    """
    query = state["query"]
    docs = state["retrieved_docs"]

    if not docs:
        return {
            **state,
            "answer": "I could not find relevant information to answer your question.",
            "citations": [],
        }

    # Build numbered context blocks
    context_blocks: List[str] = []
    sources: List[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", f"Document {i}")
        sources.append(src)
        context_blocks.append(f"[{i}] {doc.page_content}")

    context = "\n\n".join(context_blocks)

    system = (
        "You are a precise research assistant. Answer questions based strictly on the "
        "provided numbered context. Cite sources inline using [N] notation. "
        "If the context is insufficient, say so explicitly."
    )

    answer = _gemini(
        prompt=f"Context:\n{context}\n\nQuestion: {query}",
        system=system,
        max_tokens=1024,
    )

    # Extract cited source indices
    cited_nums = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    citations = [sources[i - 1] for i in sorted(cited_nums) if 1 <= i <= len(sources)]

    return {**state, "answer": answer, "citations": citations}


# ── Node: Evaluation ──────────────────────────────────────────────────────────

def eval_node(state: AgentState) -> AgentState:
    """
    Score the answer with RAGAS metrics and log everything to MLflow.
    """
    query = state["query"]
    answer = state["answer"]
    docs = state["retrieved_docs"]
    contexts = [doc.page_content for doc in docs]

    scores = _evaluator.evaluate(question=query, answer=answer, contexts=contexts)

    try:
        _logger.log_query(
            query=query,
            query_type=state["query_type"],
            answer=answer,
            eval_scores=scores,
            num_docs=len(docs),
            retry_count=state.get("retry_count", 0),
            used_web=state.get("use_web", False),
        )
    except Exception:
        pass  # logging failure must never crash the pipeline

    return {**state, "eval_scores": scores}


# ── Node: Re-retrieval ────────────────────────────────────────────────────────

def reretrieval_node(state: AgentState) -> AgentState:
    """
    Decide how to recover from a low evaluation score:
      - First retry  → switch to web search
      - Later retries → fall back to regular retrieval with use_web cleared
    """
    retry_count = state.get("retry_count", 0) + 1
    use_web = retry_count == 1  # web on first retry only
    return {**state, "retry_count": retry_count, "use_web": use_web}


# ── Edge condition ────────────────────────────────────────────────────────────

def _should_retry(state: AgentState) -> str:
    if state.get("retry_count", 0) >= settings.max_retries:
        return "end"
    if not _evaluator.passes_threshold(state.get("eval_scores", {})):
        return "retry"
    return "end"


# ── Graph assembly ────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("generation", generation_node)
    g.add_node("evaluation", eval_node)
    g.add_node("reretrieval", reretrieval_node)

    g.set_entry_point("router")
    g.add_edge("router", "retrieval")
    g.add_edge("retrieval", "generation")
    g.add_edge("generation", "evaluation")
    g.add_conditional_edges(
        "evaluation",
        _should_retry,
        {"retry": "reretrieval", "end": END},
    )
    g.add_edge("reretrieval", "retrieval")

    return g.compile()


_graph = None


def get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


# ── Public API ────────────────────────────────────────────────────────────────

def run_query(query: str) -> Dict[str, Any]:
    """
    Run a query through the full agentic RAG pipeline.

    Returns:
        answer       : str
        citations    : List[str]
        query_type   : "simple" | "multi_hop" | "web"
        eval_scores  : {"faithfulness": float, "context_precision": float, ...}
        retry_count  : int
        used_web     : bool
    """
    initial: AgentState = {
        "query": query,
        "query_type": "simple",
        "retrieved_docs": [],
        "answer": "",
        "citations": [],
        "eval_scores": {},
        "retry_count": 0,
        "use_web": False,
        "error": None,
    }
    final = get_graph().invoke(initial)
    return {
        "answer": final["answer"],
        "citations": final["citations"],
        "query_type": final["query_type"],
        "eval_scores": final["eval_scores"],
        "retry_count": final["retry_count"],
        "used_web": final["use_web"],
    }
