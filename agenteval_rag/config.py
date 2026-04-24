from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Gemini ───────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── Paths ────────────────────────────────────────────────────────────────
    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    faiss_index_path: Path = Path("data/processed/faiss_index")
    bm25_index_path: Path = Path("data/processed/bm25_index.pkl")

    # ── Retrieval ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 10
    top_k_rerank: int = 5

    # ── Evaluation thresholds ────────────────────────────────────────────────
    faithfulness_threshold: float = 0.7
    context_precision_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7
    max_retries: int = 2

    # ── Web search (optional) ────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── MLflow ───────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "agenteval_rag"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
