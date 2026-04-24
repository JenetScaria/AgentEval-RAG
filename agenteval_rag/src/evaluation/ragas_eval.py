"""
RAGAS-based evaluator: faithfulness, context_precision, answer_relevancy.

Falls back to heuristic scoring if RAGAS / its LLM dependencies are unavailable.
"""
from typing import Dict, List

from config import settings


class RAGASEvaluator:
    def __init__(self):
        self.faithfulness_threshold = settings.faithfulness_threshold
        self.context_precision_threshold = settings.context_precision_threshold
        self.answer_relevancy_threshold = settings.answer_relevancy_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str = "",
    ) -> Dict[str, float]:
        """
        Compute RAGAS metrics for one QA sample.

        Returns a dict with keys:
            faithfulness, context_precision, answer_relevancy, overall
        """
        scores = self._ragas_scores(question, answer, contexts, ground_truth)
        scores["overall"] = sum(scores.values()) / len(scores)
        return scores

    def passes_threshold(self, scores: Dict[str, float]) -> bool:
        """Return True when all three core metrics meet their thresholds."""
        return (
            scores.get("faithfulness", 0.0) >= self.faithfulness_threshold
            and scores.get("context_precision", 0.0) >= self.context_precision_threshold
            and scores.get("answer_relevancy", 0.0) >= self.answer_relevancy_threshold
        )

    # ── RAGAS scoring ─────────────────────────────────────────────────────────

    def _ragas_scores(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> Dict[str, float]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )

            dataset = Dataset.from_dict(
                {
                    "question": [question],
                    "answer": [answer],
                    "contexts": [contexts],
                    "ground_truth": [ground_truth or ""],
                }
            )
            result = evaluate(
                dataset,
                metrics=[faithfulness, context_precision, answer_relevancy],
            )
            return {
                "faithfulness": float(result["faithfulness"]),
                "context_precision": float(result["context_precision"]),
                "answer_relevancy": float(result["answer_relevancy"]),
            }
        except Exception:
            return self._heuristic_scores(question, answer, contexts)

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_scores(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        """
        Token-overlap heuristics used when RAGAS is unavailable.
        These are rough proxies, not calibrated metrics.
        """
        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        question_words = set(question.lower().split())
        context_words = set(context_text.split())

        def overlap(a: set, b: set) -> float:
            return len(a & b) / max(len(a), 1)

        return {
            "faithfulness": min(overlap(answer_words, context_words), 1.0),
            "context_precision": min(overlap(question_words, context_words), 1.0),
            "answer_relevancy": min(overlap(question_words, answer_words), 1.0),
        }
