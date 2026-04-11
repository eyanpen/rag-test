"""Retrieval evaluator: ranks candidates by cosine similarity and computes IR metrics."""

import math
from typing import Dict, List


class RetrievalEvaluator:
    """Ranks candidates by cosine similarity and computes IR metrics."""

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def rank_candidates(
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        relevant_index: int,
    ) -> int:
        """Return 1-based rank of the relevant document among candidates.

        Candidates are ranked by descending cosine similarity to the query.
        The returned rank is the 1-based position of the candidate at
        ``relevant_index`` in that sorted order.
        """
        similarities = [
            RetrievalEvaluator._cosine_similarity(query_embedding, cand)
            for cand in candidate_embeddings
        ]
        relevant_sim = similarities[relevant_index]

        # Count how many candidates have strictly higher similarity
        rank = 1
        for i, sim in enumerate(similarities):
            if i != relevant_index and sim > relevant_sim:
                rank += 1
        return rank

    @staticmethod
    def compute_metrics(
        ranks: List[int],
        total_candidates_per_query: List[int],
    ) -> Dict[str, float]:
        """Compute MRR, Recall@1, Recall@5, NDCG@10 from a list of 1-based ranks.

        Args:
            ranks: List of 1-based ranks for the relevant document per query.
            total_candidates_per_query: Number of candidates per query (unused
                for current metric definitions but kept for interface consistency).

        Returns:
            {"mrr": float, "recall_at_1": float, "recall_at_5": float, "ndcg_at_10": float}
        """
        if not ranks:
            return {"mrr": 0.0, "recall_at_1": 0.0, "recall_at_5": 0.0, "ndcg_at_10": 0.0}

        n = len(ranks)

        # MRR: Mean Reciprocal Rank
        mrr = sum(1.0 / r for r in ranks) / n

        # Recall@K: fraction of queries where relevant doc is in top-K
        recall_at_1 = sum(1 for r in ranks if r <= 1) / n
        recall_at_5 = sum(1 for r in ranks if r <= 5) / n

        # NDCG@10 with binary relevance (single relevant doc)
        # DCG@10 = 1/log2(rank+1) if rank <= 10, else 0
        # IDCG@10 = 1/log2(2) = 1.0 (ideal: relevant doc at rank 1)
        # NDCG@10 = DCG@10 / IDCG@10
        idcg = 1.0 / math.log2(2)  # = 1.0
        ndcg_at_10 = sum(
            (1.0 / math.log2(r + 1)) / idcg for r in ranks if r <= 10
        ) / n

        return {
            "mrr": mrr,
            "recall_at_1": recall_at_1,
            "recall_at_5": recall_at_5,
            "ndcg_at_10": ndcg_at_10,
        }

    @staticmethod
    def compute_metrics_by_difficulty(
        ranks: List[int],
        total_candidates_per_query: List[int],
        difficulties: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics grouped by difficulty level.

        Args:
            ranks: List of 1-based ranks per query.
            total_candidates_per_query: Number of candidates per query.
            difficulties: Difficulty label per query ("easy", "medium", "hard").

        Returns:
            {"easy": {...}, "medium": {...}, "hard": {...}, "overall": {...}}
            Each value is a dict with mrr, recall_at_1, recall_at_5, ndcg_at_10.
        """
        result: Dict[str, Dict[str, float]] = {}

        for level in ("easy", "medium", "hard"):
            level_ranks = []
            level_candidates = []
            for r, c, d in zip(ranks, total_candidates_per_query, difficulties):
                if d == level:
                    level_ranks.append(r)
                    level_candidates.append(c)
            result[level] = RetrievalEvaluator.compute_metrics(level_ranks, level_candidates)

        result["overall"] = RetrievalEvaluator.compute_metrics(ranks, total_candidates_per_query)
        return result
