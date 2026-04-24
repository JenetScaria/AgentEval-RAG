"""
Streamlit UI for AgentEval RAG.

Run:
    streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="AgentEval RAG",
    page_icon="🔍",
    layout="wide",
)

# Lazy import so startup is fast even if indexes aren't built yet
@st.cache_resource(show_spinner="Loading pipeline …")
def load_graph():
    from graph import get_graph, run_query  # noqa: F401

    get_graph()
    return run_query


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ AgentEval RAG")
    st.markdown(
        """
        **Self-evaluating agentic RAG pipeline**

        - **Router** classifies query type
        - **Hybrid Retrieval** (FAISS + BM25 + reranker)
        - **Generation** via Claude with citations
        - **RAGAS Evaluation** scores faithfulness, precision, relevancy
        - Auto-retries with web fallback when score < threshold
        """
    )
    st.divider()
    st.caption("Thresholds (see config.py)")
    from config import settings

    st.metric("Faithfulness", f"{settings.faithfulness_threshold:.0%}")
    st.metric("Context Precision", f"{settings.context_precision_threshold:.0%}")
    st.metric("Answer Relevancy", f"{settings.answer_relevancy_threshold:.0%}")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🔍 AgentEval RAG — Ask your documents")

query = st.text_input(
    "Enter your question",
    placeholder="e.g. What is the transformer attention mechanism?",
)

if st.button("Ask", type="primary") and query.strip():
    run_query = load_graph()

    with st.spinner("Thinking …"):
        try:
            result = run_query(query)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("Answer")
    st.markdown(result["answer"])

    # ── Citations ─────────────────────────────────────────────────────────────
    if result["citations"]:
        with st.expander("📚 Citations", expanded=False):
            for i, src in enumerate(result["citations"], 1):
                st.write(f"{i}. `{src}`")

    # ── Metadata ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Query type", result["query_type"].replace("_", " ").title())
    with col2:
        st.metric("Retry count", result["retry_count"])
    with col3:
        st.metric("Web fallback", "Yes" if result["used_web"] else "No")

    # ── Eval scores ───────────────────────────────────────────────────────────
    scores = result["eval_scores"]
    if scores:
        st.subheader("📊 Evaluation Scores")
        score_cols = st.columns(len(scores))
        for col, (metric, value) in zip(score_cols, scores.items()):
            col.metric(metric.replace("_", " ").title(), f"{value:.2f}")

        overall = scores.get("overall", 0.0)
        colour = "green" if overall >= 0.7 else "orange" if overall >= 0.5 else "red"
        st.markdown(
            f"**Overall quality:** :{colour}[{overall:.2f}]"
        )

elif query.strip() == "" and st.session_state.get("_asked"):
    st.warning("Please enter a question.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AgentEval RAG · powered by Claude (claude-opus-4-6) · "
    "LangGraph · FAISS · BM25 · RAGAS · MLflow"
)
