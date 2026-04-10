"""Benchmark report generation (Markdown + JSON)."""
import math
from typing import Any, Dict

from models import BenchmarkSummary


class ReportGenerator:
    @staticmethod
    def generate_markdown(summary: BenchmarkSummary) -> str:
        lines = ["# Embedding 模型基准测试报告\n"]
        lines.append(f"> 生成时间: {summary.end_time}\n")

        # 1. 测试环境信息
        lines.append("## 1. 测试环境信息\n")
        lines.append("| 项目 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| API 地址 | {summary.config.api_base_url} |")
        lines.append(f"| LLM 模型 | {summary.config.llm_model} |")
        lines.append(f"| FalkorDB | {summary.config.falkordb_host}:{summary.config.falkordb_port} |")
        lines.append(f"| 数据集 | {', '.join(summary.config.datasets)} |")
        lines.append(f"| 采样数量 | {summary.config.sample_size} |")
        lines.append(f"| 总耗时 | {summary.total_time_seconds:.1f}s |")
        lines.append("")

        # 2. 模型清单
        lines.append("## 2. 模型清单\n")
        lines.append("| 模型 | 维度 | 最大Token | 状态 |")
        lines.append("|------|------|----------|------|")
        for m in summary.config.models:
            status = "✅"
            for r in summary.model_results:
                if r.model.name == m.name and r.error and "unavailable" in r.error.lower():
                    status = "❌ 不可用"
                    break
            lines.append(f"| {m.display_name} | {m.dim} | {m.max_tokens} | {status} |")
        lines.append("")

        # 3. Phase 1-9 共享耗时
        if summary.dataset_phases:
            lines.append("## 3. Phase 1-9 共享耗时\n")
            lines.append("| 数据集 | 耗时 | 状态 |")
            lines.append("|--------|------|------|")
            for dp in summary.dataset_phases:
                status = "✅" if not dp.error else f"❌ {dp.error}"
                lines.append(f"| {dp.dataset_name} | {dp.phase_1_9_time_seconds:.1f}s | {status} |")
            lines.append("")

        # 4. Per-dataset metrics
        ds_names = sorted(set(r.dataset_name for r in summary.model_results))
        for ds in ds_names:
            ds_results = [r for r in summary.model_results if r.dataset_name == ds and r.evaluation]
            if not ds_results:
                continue
            lines.append(f"## 4. 指标对比表 — {ds}\n")
            lines.append("| 模型 | ROUGE-L | Answer Correctness | Phase 10 耗时 | 查询耗时 |")
            lines.append("|------|---------|-------------------|-------------|---------|")
            rouges = [r.evaluation.rouge_l for r in ds_results if r.evaluation and not math.isnan(r.evaluation.rouge_l)]
            accs = [r.evaluation.answer_correctness for r in ds_results if r.evaluation and not math.isnan(r.evaluation.answer_correctness)]
            best_rouge = max(rouges) if rouges else None
            best_acc = max(accs) if accs else None
            for r in ds_results:
                ev = r.evaluation
                rv = f"{ev.rouge_l:.4f}" if not math.isnan(ev.rouge_l) else "NaN"
                av = f"{ev.answer_correctness:.4f}" if not math.isnan(ev.answer_correctness) else "NaN"
                if best_rouge is not None and not math.isnan(ev.rouge_l) and ev.rouge_l == best_rouge:
                    rv = f"**{rv}**"
                if best_acc is not None and not math.isnan(ev.answer_correctness) and ev.answer_correctness == best_acc:
                    av = f"**{av}**"
                lines.append(f"| {r.model.display_name} | {rv} | {av} | {r.embedding_time_seconds:.1f}s | {r.query_time_seconds:.1f}s |")
            lines.append("")

        # 5. BGE-M3 模式对比
        bge_results = [r for r in summary.model_results if r.model.name.startswith("BAAI/bge-m3") and r.evaluation]
        if bge_results:
            lines.append("## 5. BGE-M3 模式对比\n")
            lines.append("| 模式 | 数据集 | ROUGE-L | Answer Correctness |")
            lines.append("|------|--------|---------|-------------------|")
            rouges = [r.evaluation.rouge_l for r in bge_results if not math.isnan(r.evaluation.rouge_l)]
            accs = [r.evaluation.answer_correctness for r in bge_results if not math.isnan(r.evaluation.answer_correctness)]
            br = max(rouges) if rouges else None
            ba = max(accs) if accs else None
            for r in bge_results:
                ev = r.evaluation
                rv = f"{ev.rouge_l:.4f}" if not math.isnan(ev.rouge_l) else "NaN"
                av = f"{ev.answer_correctness:.4f}" if not math.isnan(ev.answer_correctness) else "NaN"
                if br and not math.isnan(ev.rouge_l) and ev.rouge_l == br:
                    rv = f"**{rv}**"
                if ba and not math.isnan(ev.answer_correctness) and ev.answer_correctness == ba:
                    av = f"**{av}**"
                lines.append(f"| {r.model.display_name} | {r.dataset_name} | {rv} | {av} |")
            lines.append("")

        # 6. 总体排名
        lines.append("## 6. 总体排名\n")
        model_scores: Dict[str, list] = {}
        for r in summary.model_results:
            if r.evaluation:
                vals = []
                if not math.isnan(r.evaluation.rouge_l):
                    vals.append(r.evaluation.rouge_l)
                if not math.isnan(r.evaluation.answer_correctness):
                    vals.append(r.evaluation.answer_correctness)
                if vals:
                    model_scores.setdefault(r.model.display_name, []).extend(vals)
        ranked = sorted(model_scores.items(), key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0, reverse=True)
        lines.append("| 排名 | 模型 | 加权平均分 |")
        lines.append("|------|------|----------|")
        for i, (name, scores) in enumerate(ranked, 1):
            avg = sum(scores) / len(scores) if scores else 0
            lines.append(f"| {i} | {name} | {avg:.4f} |")
        lines.append("")

        # 7. 耗时统计
        lines.append("## 7. 耗时统计\n")
        lines.append("| 模型 | 数据集 | Phase 10 | 查询时间 | 评估时间 | 总时间 |")
        lines.append("|------|--------|---------|---------|---------|--------|")
        for r in summary.model_results:
            et = r.evaluation.eval_time_seconds if r.evaluation else 0
            total = r.embedding_time_seconds + r.query_time_seconds + et
            lines.append(
                f"| {r.model.display_name} | {r.dataset_name} "
                f"| {r.embedding_time_seconds:.1f}s | {r.query_time_seconds:.1f}s "
                f"| {et:.1f}s | {total:.1f}s |"
            )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def generate_summary_json(summary: BenchmarkSummary) -> dict:
        results = []
        for r in summary.model_results:
            entry: Dict[str, Any] = {
                "model": r.model.display_name,
                "model_name": r.model.name,
                "dataset": r.dataset_name,
                "error": r.error,
                "embedding_time": r.embedding_time_seconds,
                "query_time": r.query_time_seconds,
                "num_predictions": len(r.predictions),
            }
            if r.evaluation:
                entry["rouge_l"] = r.evaluation.rouge_l
                entry["answer_correctness"] = r.evaluation.answer_correctness
                entry["eval_time"] = r.evaluation.eval_time_seconds
            results.append(entry)

        dataset_phases = []
        for dp in summary.dataset_phases:
            dataset_phases.append({
                "dataset": dp.dataset_name,
                "phase_1_9_time": dp.phase_1_9_time_seconds,
                "output_dir": dp.output_dir,
                "error": dp.error,
            })

        return {
            "config": {
                "api_base_url": summary.config.api_base_url,
                "llm_model": summary.config.llm_model,
                "falkordb": f"{summary.config.falkordb_host}:{summary.config.falkordb_port}",
                "sample_size": summary.config.sample_size,
                "datasets": summary.config.datasets,
                "models": [m.name for m in summary.config.models],
            },
            "start_time": summary.start_time,
            "end_time": summary.end_time,
            "total_time_seconds": summary.total_time_seconds,
            "dataset_phases": dataset_phases,
            "results": results,
        }
