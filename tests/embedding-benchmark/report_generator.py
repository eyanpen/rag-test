"""LLM-powered Markdown report generator for embedding benchmark results."""

import json
import logging
import requests
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates LLM-analyzed Markdown report from benchmark results."""

    def __init__(self, api_base: str, model: str, timeout: int = 300):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_report(self, results: dict) -> str:
        """Send results to LLM, return Markdown report.

        Falls back to template report if LLM call fails or times out.
        """
        try:
            return self._call_llm(results)
        except Exception as e:
            logger.warning("LLM report generation failed: %s. Using fallback template.", e)
            return self.generate_fallback_report(results)

    def _call_llm(self, results: dict) -> str:
        """Call the LLM API to generate an analytical report."""
        results_json = json.dumps(results, indent=2, default=str)

        prompt = (
            "You are an expert in embedding models and information retrieval evaluation. "
            "Analyze the following benchmark results and produce a structured Markdown report.\n\n"
            "The benchmark evaluates embedding models in two modes:\n"
            "- Symmetric mode: query and documents embedded without any prefix\n"
            "- Dual-tower mode: query embedded with query_prefix, documents with document_prefix\n\n"
            "Metrics used: MRR (Mean Reciprocal Rank), Recall@1, Recall@5, NDCG@10.\n\n"
            "Your report MUST include these sections in order:\n"
            "1. **Executive Summary** — Key findings and top-performing model/mode combinations\n"
            "2. **Methodology** — Brief description of the evaluation approach\n"
            "3. **Results Table** — All models × all metrics in a Markdown table\n"
            "4. **Dual-Tower vs Symmetric Analysis** — Compare modes for models that support both, "
            "using the delta values provided\n"
            "5. **Per-Difficulty Breakdown** — How models perform across easy/medium/hard queries\n"
            "6. **Model Rankings** — Rank models by overall retrieval quality\n"
            "7. **Recommendations** — Which model/mode to use for different use cases\n\n"
            "Here are the benchmark results:\n\n"
            f"```json\n{results_json}\n```"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a technical report writer specializing in ML model evaluation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        url = f"{self.api_base}/chat/completions"
        logger.info("Calling LLM API at %s for report generation (timeout=%ds)...", url, self.timeout)

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return content

    @staticmethod
    def generate_fallback_report(results: dict) -> str:
        """Generate basic template report without LLM analysis."""
        lines = ["# Embedding Model Retrieval Benchmark Report\n"]
        lines.append(f"> Generated: {results.get('timestamp', datetime.now().isoformat())}\n")

        # --- Dataset Stats ---
        stats = results.get("dataset_stats", {})
        if stats:
            lines.append("## Methodology\n")
            lines.append(f"- Total evaluation items: {stats.get('total_items', 'N/A')}")
            lines.append(f"- Domains: {', '.join(stats.get('domains', []))}")
            diff_dist = stats.get("difficulty_distribution", {})
            if diff_dist:
                parts = [f"{k}: {v}" for k, v in diff_dist.items()]
                lines.append(f"- Difficulty distribution: {', '.join(parts)}")
            lines.append("")

        # --- Results Table ---
        model_results = results.get("results", [])
        if model_results:
            lines.append("## Results Table\n")
            lines.append("| Rank | Model | Mode | MRR | Recall@1 | Recall@5 | NDCG@10 | Time (s) |")
            lines.append("|------|-------|------|-----|----------|----------|---------|----------|")
            for i, r in enumerate(model_results, 1):
                metrics = r.get("metrics", {})
                lines.append(
                    f"| {i} "
                    f"| {r.get('model_display_name', 'N/A')} "
                    f"| {r.get('mode', 'N/A')} "
                    f"| {metrics.get('mrr', 0):.4f} "
                    f"| {metrics.get('recall_at_1', 0):.4f} "
                    f"| {metrics.get('recall_at_5', 0):.4f} "
                    f"| {metrics.get('ndcg_at_10', 0):.4f} "
                    f"| {r.get('embedding_time_seconds', 0):.1f} |"
                )
            lines.append("")

        # --- Dual-Tower vs Symmetric Deltas ---
        deltas = results.get("deltas", [])
        if deltas:
            lines.append("## Dual-Tower vs Symmetric Analysis\n")
            lines.append("| Model | ΔMRR | ΔRecall@1 | ΔRecall@5 | ΔNDCG@10 |")
            lines.append("|-------|------|-----------|-----------|----------|")
            for d in deltas:
                lines.append(
                    f"| {d.get('model_display_name', 'N/A')} "
                    f"| {d.get('delta_mrr', 0):+.4f} "
                    f"| {d.get('delta_recall_at_1', 0):+.4f} "
                    f"| {d.get('delta_recall_at_5', 0):+.4f} "
                    f"| {d.get('delta_ndcg_at_10', 0):+.4f} |"
                )
            lines.append("")

        # --- Per-Difficulty Breakdown ---
        has_difficulty = any(r.get("metrics_by_difficulty") for r in model_results)
        if has_difficulty:
            lines.append("## Per-Difficulty Breakdown\n")
            for difficulty in ["easy", "medium", "hard"]:
                lines.append(f"### {difficulty.capitalize()}\n")
                lines.append("| Model | Mode | MRR | Recall@1 | Recall@5 | NDCG@10 |")
                lines.append("|-------|------|-----|----------|----------|---------|")
                for r in model_results:
                    diff_metrics = r.get("metrics_by_difficulty", {}).get(difficulty, {})
                    if diff_metrics:
                        lines.append(
                            f"| {r.get('model_display_name', 'N/A')} "
                            f"| {r.get('mode', 'N/A')} "
                            f"| {diff_metrics.get('mrr', 0):.4f} "
                            f"| {diff_metrics.get('recall_at_1', 0):.4f} "
                            f"| {diff_metrics.get('recall_at_5', 0):.4f} "
                            f"| {diff_metrics.get('ndcg_at_10', 0):.4f} |"
                        )
                lines.append("")

        # --- Model Rankings ---
        completed = [r for r in model_results if r.get("status") == "completed"]
        if completed:
            lines.append("## Model Rankings (by MRR)\n")
            sorted_results = sorted(completed, key=lambda r: r.get("metrics", {}).get("mrr", 0), reverse=True)
            lines.append("| Rank | Model | Mode | MRR |")
            lines.append("|------|-------|------|-----|")
            for i, r in enumerate(sorted_results, 1):
                lines.append(
                    f"| {i} "
                    f"| {r.get('model_display_name', 'N/A')} "
                    f"| {r.get('mode', 'N/A')} "
                    f"| {r.get('metrics', {}).get('mrr', 0):.4f} |"
                )
            lines.append("")

        # --- Skipped Models ---
        skipped = results.get("skipped_models", [])
        if skipped:
            lines.append("## Skipped Models\n")
            lines.append("| Model | Reason |")
            lines.append("|-------|--------|")
            for s in skipped:
                lines.append(f"| {s.get('model', 'N/A')} | {s.get('reason', 'N/A')} |")
            lines.append("")

        # --- Recommendations placeholder ---
        lines.append("## Recommendations\n")
        if completed:
            best = max(completed, key=lambda r: r.get("metrics", {}).get("mrr", 0))
            lines.append(
                f"Based on MRR, the top-performing configuration is "
                f"**{best.get('model_display_name', 'N/A')}** in "
                f"**{best.get('mode', 'N/A')}** mode "
                f"(MRR: {best.get('metrics', {}).get('mrr', 0):.4f}).\n"
            )
            if deltas:
                positive_deltas = [d for d in deltas if d.get("delta_mrr", 0) > 0]
                if positive_deltas:
                    lines.append(
                        "Dual-tower mode shows improvement for: "
                        + ", ".join(d.get("model_display_name", "N/A") for d in positive_deltas)
                        + ".\n"
                    )
                else:
                    lines.append("No models showed MRR improvement with dual-tower mode.\n")
        else:
            lines.append("No models completed evaluation. Check skipped models above for details.\n")

        return "\n".join(lines)
