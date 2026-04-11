#!/usr/bin/env python3
"""All-model symmetric vs dual-tower embedding comparison benchmark."""

import json
import numpy as np
import sys, os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from embedding_client import EmbeddingClient, EmbeddingAPIError
from models import EMBEDDING_MODELS

API_BASE = "http://10.210.156.69:8633"

QUERIES = [
    ("首都", "What is the capital of China?"),
    ("引力", "Explain gravity"),
]
DOCUMENTS = [
    ("北京", "The capital of China is Beijing."),
    ("引力", "Gravity is a force that attracts two bodies towards each other. "
             "It gives weight to physical objects and is responsible for the "
             "movement of planets around the sun."),
]


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return a @ b.T / (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T)


def eval_mode(client, model_name, prefix_q="", prefix_d=""):
    q_embs = client.get_embeddings(model_name, [q for _, q in QUERIES], prefix=prefix_q)
    d_embs = client.get_embeddings(model_name, [d for _, d in DOCUMENTS], prefix=prefix_d)
    return cosine_similarity(q_embs, d_embs)


def fmt_sim_table(title, sim):
    """Format a similarity matrix as a readable text table."""
    n_q, n_d = sim.shape
    # Header
    doc_headers = "".join(f"  Doc {j} ({DOCUMENTS[j][0]})" for j in range(n_d))
    lines = [title, " " * 16 + doc_headers]
    for i in range(n_q):
        best = int(np.argmax(sim[i]))
        mark = "✓" if best == i else "✗"
        vals = "".join(f"  {sim[i][j]:>12.4f}   " for j in range(n_d))
        lines.append(f"Query {i} ({QUERIES[i][0]:>2s}){vals}  {mark}")
    return "\n".join(lines)


def fmt_delta_table(title, delta):
    n_q, n_d = delta.shape
    doc_headers = "".join(f"  Doc {j}       " for j in range(n_d))
    lines = [title, " " * 16 + doc_headers]
    for i in range(n_q):
        vals = "".join(f"  {delta[i][j]:>+12.4f}   " for j in range(n_d))
        lines.append(f"Query {i} ({QUERIES[i][0]:>2s}){vals}")
    return "\n".join(lines)


def main():
    client = EmbeddingClient(api_base=API_BASE)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report_lines = [
        f"# Symmetric vs Dual-Tower Embedding Comparison",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for model in EMBEDDING_MODELS:
        header = f"{'='*60}\n  {model.display_name} ({model.name})\n{'='*60}"
        print(f"\n{header}")

        # Health check
        try:
            client.get_embeddings(model.name, ["hello"])
        except Exception as e:
            msg = f"  ❌ SKIP: {e}"
            print(msg)
            report_lines += [header, msg, ""]
            continue

        sim_sym = eval_mode(client, model.name)
        sym_text = fmt_sim_table("=== Symmetric (no prefix) ===", sim_sym)
        print(sym_text)

        block = [header, sym_text]

        if model.supports_dual_tower:
            sim_dt = eval_mode(client, model.name, model.query_prefix, model.document_prefix)
            dt_text = fmt_sim_table("=== Dual-Tower (with prefix) ===", sim_dt)
            delta_text = fmt_delta_table("=== Delta (dual-tower - symmetric) ===", sim_dt - sim_sym)
            print(dt_text)
            print(delta_text)
            block += ["", dt_text, "", delta_text]

        report_lines += block + ["", ""]

    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n✅ Report saved to {report_path}")


if __name__ == "__main__":
    main()
