#!/usr/bin/env python3
"""测试各 Embedding 模型的双塔（Bi-Encoder）支持情况。

用一个真实的 query + document 对测试：
- query: "What is fiber optic communication?"
- document: tests/fiber_optic_communication.md 的内容

对每个模型分别用 bare（无 prefix）、query prefix、document prefix 编码，
比较向量余弦相似度。如果 prefix 导致向量显著不同（cos < 0.999），说明模型支持双塔。

同时测试双塔模式下 query-document 的检索相关性是否优于对称模式。

用法:
  python tests/test_dual_tower_support.py
  python -m pytest tests/test_dual_tower_support.py -v
"""
import os
import httpx
import numpy as np
import pytest

API_BASE = "http://10.210.156.69:8633"
TIMEOUT = 60

QUERY_TEXT = "What is fiber optic communication?"
DOC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fiber_optic_communication.md")

# (model_name, query_prefix, document_prefix)
MODEL_CONFIGS = [
    ("BAAI/bge-m3", "query: ", "passage: "),
    ("BAAI/bge-m3/heavy", "query: ", "passage: "),
    ("BAAI/bge-m3/interactive", "query: ", "passage: "),
    (
        "intfloat/e5-mistral-7b-instruct",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "",
    ),
    ("intfloat/multilingual-e5-large-instruct", "query: ", "passage: "),
    ("nomic-ai/nomic-embed-text-v1.5", "search_query: ", "search_document: "),
    (
        "Qwen/Qwen3-Embedding-8B",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "",
    ),
    (
        "Qwen/Qwen3-Embedding-8B-Alt",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "",
    ),
]


# ── Helpers ──


def _api_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE}/health", timeout=5)
        return True
    except Exception:
        return False


def _load_document() -> str:
    with open(DOC_PATH, "r", encoding="utf-8") as f:
        return f.read()


def _embed(model: str, texts: list[str]) -> list[list[float]]:
    print(f"Embedding {len(texts)} text({texts[0][:80]}) with {model}")
    r = httpx.post(
        f"{API_BASE}/embeddings",
        json={"model": model, "input": texts},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()["data"]
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def _cosine(a, b) -> float:
    a, b = np.array(a), np.array(b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


skip_if_unreachable = pytest.mark.skipif(
    not _api_reachable(),
    reason=f"API {API_BASE} is not reachable",
)


# ── Tests ──


@skip_if_unreachable
@pytest.mark.parametrize("model,qp,dp", MODEL_CONFIGS, ids=[c[0].split("/")[-1] for c in MODEL_CONFIGS])
def test_prefix_changes_vector(model, qp, dp):
    """Verify that adding query/document prefix produces a different vector (cos < 0.999)."""
    doc_text = _load_document()

    try:
        bare_q = _embed(model, [QUERY_TEXT])[0]
        bare_d = _embed(model, [doc_text])[0]
    except httpx.HTTPStatusError as e:
        pytest.skip(f"Model {model} unavailable: {e.response.status_code}")

    if qp:
        prefixed_q = _embed(model, [qp + QUERY_TEXT])[0]
        cos_q = _cosine(bare_q, prefixed_q)
        assert cos_q < 0.999, f"Query prefix had no effect: cos={cos_q:.6f}"

    if dp:
        prefixed_d = _embed(model, [dp + doc_text])[0]
        cos_d = _cosine(bare_d, prefixed_d)
        assert cos_d < 0.999, f"Document prefix had no effect: cos={cos_d:.6f}"


@skip_if_unreachable
@pytest.mark.parametrize("model,qp,dp", MODEL_CONFIGS, ids=[c[0].split("/")[-1] for c in MODEL_CONFIGS])
def test_dual_tower_retrieval_relevance(model, qp, dp):
    """Compare query-document similarity in symmetric vs dual-tower mode.

    In dual-tower mode (with proper prefixes), the query-document similarity
    should ideally be higher than bare symmetric mode, indicating better
    retrieval alignment.
    """
    doc_text = _load_document()

    try:
        bare_q = _embed(model, [QUERY_TEXT])[0]
        bare_d = _embed(model, [doc_text])[0]
    except httpx.HTTPStatusError as e:
        pytest.skip(f"Model {model} unavailable: {e.response.status_code}")

    cos_symmetric = _cosine(bare_q, bare_d)

    # Dual-tower: query with query_prefix, document with document_prefix
    dt_q = _embed(model, [qp + QUERY_TEXT])[0] if qp else bare_q
    dt_d = _embed(model, [dp + doc_text])[0] if dp else bare_d
    cos_dual_tower = _cosine(dt_q, dt_d)

    # Report both scores (test always passes, this is informational)
    print(f"\n  {model}:")
    print(f"    Symmetric (bare q ↔ bare d):       {cos_symmetric:.6f}")
    print(f"    Dual-tower (prefix q ↔ prefix d):  {cos_dual_tower:.6f}")
    print(f"    Delta:                              {cos_dual_tower - cos_symmetric:+.6f}")


# ── Standalone runner ──


def main():
    """Run all tests as a standalone script with formatted output."""
    doc_text = _load_document()
    doc_preview = doc_text[:80].replace("\n", " ") + "..."

    print(f"Query:    {QUERY_TEXT}")
    print(f"Document: {doc_preview}")
    print(f"Doc len:  {len(doc_text)} chars")
    print()

    header = (
        f"{'Model':<35} "
        f"{'cos(bare_q, bare_d)':<22} "
        f"{'cos(dt_q, dt_d)':<18} "
        f"{'Delta':<10} "
        f"{'cos(bare_q, dt_q)':<20} "
        f"{'cos(bare_d, dt_d)':<20} "
        f"DualTower?"
    )
    print(header)
    print("-" * len(header))

    for model, qp, dp in MODEL_CONFIGS:
        try:
            bare_q = _embed(model, [QUERY_TEXT])[0]
            bare_d = _embed(model, [doc_text])[0]
            cos_sym = _cosine(bare_q, bare_d)

            dt_q = _embed(model, [qp + QUERY_TEXT])[0] if qp else bare_q
            dt_d = _embed(model, [dp + doc_text])[0] if dp else bare_d
            cos_dt = _cosine(dt_q, dt_d)

            cos_qq = _cosine(bare_q, dt_q) if qp else 1.0
            cos_dd = _cosine(bare_d, dt_d) if dp else 1.0

            delta = cos_dt - cos_sym
            is_dt = cos_qq < 0.999 or cos_dd < 0.999

            print(
                f"{model:<35} "
                f"{cos_sym:<22.6f} "
                f"{cos_dt:<18.6f} "
                f"{delta:<+10.6f} "
                f"{cos_qq:<20.6f} "
                f"{cos_dd:<20.6f} "
                f"{'YES' if is_dt else 'NO'}"
            )
        except httpx.HTTPStatusError as e:
            print(f"{model:<35} ERROR: HTTP {e.response.status_code}")
        except Exception as e:
            print(f"{model:<35} ERROR: {e}")

    # BGE-M3 cross-variant check
    print("\n=== BGE-M3 cross-variant comparison (bare vectors) ===")
    bge_vecs = {}
    for m in ["BAAI/bge-m3", "BAAI/bge-m3/heavy", "BAAI/bge-m3/interactive"]:
        try:
            bge_vecs[m] = _embed(m, [QUERY_TEXT])[0]
        except Exception as e:
            print(f"  {m}: ERROR {e}")

    names = list(bge_vecs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = _cosine(bge_vecs[names[i]], bge_vecs[names[j]])
            tag = "DIFFERENT" if c < 0.999 else "IDENTICAL"
            n1 = names[i].split("/")[-1]
            n2 = names[j].split("/")[-1]
            print(f"  cos({n1:>15}, {n2:<15}) = {c:.6f}  {tag}")


if __name__ == "__main__":
    main()
