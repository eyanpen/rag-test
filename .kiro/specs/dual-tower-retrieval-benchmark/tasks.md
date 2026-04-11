# Implementation Plan: Dual-Tower Retrieval Benchmark

## Overview

Build a standalone retrieval benchmark under `tests/embedding-benchmark/` that evaluates embedding models in symmetric and dual-tower modes using IR metrics (MRR, Recall@K, NDCG@10). Four Python modules plus a curated dataset, reusing `tests/models.py` for model configs and the existing API at `http://10.210.156.69:8633`.

## Tasks

- [x] 1. Create evaluation dataset
  - [x] 1.1 Create `tests/embedding-benchmark/dataset.json` with 30+ eval items
    - 5+ domains: science, history, technology, medicine, law (and optionally more)
    - 3 difficulty levels: easy, medium, hard (roughly balanced)
    - Each item: one query, one relevant_doc, 7–19 distractor_docs, domain, difficulty
    - All text in English; distractors are topically plausible per difficulty level
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 1.2 Write property test for dataset structural validity
    - **Property 1: Eval item structural validity**
    - Validate every item has non-empty query, non-empty relevant_doc, 7–19 non-empty distractor_docs, valid domain, difficulty in {"easy", "medium", "hard"}
    - Test file: `tests/embedding-benchmark/test_benchmark.py`
    - **Validates: Requirements 1.2, 1.5, 1.6**

- [x] 2. Implement embedding client
  - [x] 2.1 Create `tests/embedding-benchmark/embedding_client.py`
    - `EmbeddingClient` class with `__init__(api_base, timeout=120)`
    - `get_embeddings(model, texts, prefix="")` method
    - Prepend prefix to each text before sending
    - Batch into groups of 64 when len(texts) > 64, sequential API calls
    - Sort response `data` array by `index` field before extracting vectors
    - Raise `EmbeddingAPIError(model, status_code, response_body)` on HTTP errors
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 2.2 Write property test for embedding response reordering
    - **Property 2: Embedding response reordering preserves input correspondence**
    - Mock API returning shuffled index values, verify sorted output matches input order
    - **Validates: Requirements 2.2**

  - [ ]* 2.3 Write property test for prefix prepending
    - **Property 3: Prefix is prepended to every input text**
    - Generate random prefix strings and text lists, verify each sent text equals prefix + original
    - **Validates: Requirements 2.3**

  - [ ]* 2.4 Write property test for HTTP error handling
    - **Property 4: HTTP errors produce descriptive exceptions**
    - Generate random HTTP error codes (400–599) and response bodies, verify exception contains model name, status code, and body
    - **Validates: Requirements 2.5**

  - [ ]* 2.5 Write property test for batching
    - **Property 5: Batching splits correctly and preserves order**
    - Generate random text lists of size 1–200, mock API, verify ceil(N/64) calls and N embeddings in order
    - **Validates: Requirements 2.6**

- [x] 3. Checkpoint — Verify dataset and embedding client
  - Ensure dataset.json is valid and embedding_client.py handles prefix, batching, sorting, and errors correctly. Ask the user if questions arise.

- [x] 4. Implement retrieval evaluator
  - [x] 4.1 Create `tests/embedding-benchmark/retrieval_evaluator.py`
    - `RetrievalEvaluator` class with static methods
    - `rank_candidates(query_embedding, candidate_embeddings, relevant_index)` → 1-based rank by descending cosine similarity
    - `compute_metrics(ranks, total_candidates_per_query)` → {"mrr", "recall_at_1", "recall_at_5", "ndcg_at_10"}
    - `compute_metrics_by_difficulty(ranks, total_candidates_per_query, difficulties)` → {"easy": {...}, "medium": {...}, "hard": {...}, "overall": {...}}
    - _Requirements: 3.3, 3.4, 3.5_

  - [ ]* 4.2 Write property test for cosine ranking
    - **Property 6: Candidates are ranked by descending cosine similarity**
    - Generate random query + candidate vectors, verify ranking order and relevant doc rank
    - **Validates: Requirements 3.3**

  - [ ]* 4.3 Write property test for IR metric computation
    - **Property 7: IR metric computation correctness**
    - Generate random rank lists, verify MRR = mean(1/rank), Recall@K = count(rank ≤ K)/len, NDCG@10 per standard formula
    - **Validates: Requirements 3.4**

  - [ ]* 4.4 Write property test for per-difficulty metrics
    - **Property 8: Per-difficulty metrics equal filtered-subset metrics**
    - Generate random (rank, difficulty) pairs, verify grouped metrics match subset computation
    - **Validates: Requirements 3.5**

- [x] 5. Implement report generator
  - [x] 5.1 Create `tests/embedding-benchmark/report_generator.py`
    - `ReportGenerator` class with `__init__(api_base, model, timeout=300)`
    - `generate_report(results)` → Markdown string via LLM `/chat/completions` call
    - LLM prompt instructs sections: Executive Summary, Methodology, Results Table, Dual-Tower vs Symmetric Analysis, Per-Difficulty Breakdown, Model Rankings, Recommendations
    - `generate_fallback_report(results)` → basic template report without LLM
    - Falls back to template on LLM failure or timeout
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 6. Checkpoint — Verify evaluator and report generator
  - Ensure retrieval_evaluator.py computes correct metrics and report_generator.py handles both LLM and fallback paths. Ask the user if questions arise.

- [x] 7. Implement benchmark runner and wire everything together
  - [x] 7.1 Create `tests/embedding-benchmark/run_benchmark.py` entry point
    - Parse CLI args: `--models` (comma-separated), `--output-dir` (default: `tests/embedding-benchmark/`)
    - Import `EMBEDDING_MODELS` and `EmbeddingModelConfig` from `tests.models`
    - Filter models by `--models` arg if provided
    - Load `dataset.json`
    - Configure logging to stdout + `benchmark.log`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 7.2 Implement per-model evaluation loop
    - Health-check each model (single short text embedding call) before full eval
    - Skip models that fail health check, add to `skipped_models` with reason
    - Symmetric mode: embed query + all candidates without prefix, rank, collect ranks
    - Dual-tower mode (if `supports_dual_tower`): embed query with `query_prefix`, candidates with `document_prefix`, rank, collect ranks
    - Compute metrics overall + per-difficulty for each mode
    - Compute delta rows (dual-tower minus symmetric) for dual-tower models
    - Record embedding time per model-mode
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.4, 6.1, 6.2, 6.3_

  - [x] 7.3 Implement results assembly and output
    - Rank all model-mode results by MRR descending
    - Build `BenchmarkResults` with results, deltas, skipped_models, timestamp, dataset_stats
    - Save `results.json`
    - Generate `report.md` via `ReportGenerator`
    - Print output file paths
    - Exit non-zero if all models failed
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.4, 6.4, 6.5, 6.6, 7.5, 7.6_

  - [ ]* 7.4 Write property test for results sorting
    - **Property 9: Results are sorted by MRR descending**
    - Generate random ModelModeResult lists, verify MRR non-increasing order
    - **Validates: Requirements 4.3**

  - [ ]* 7.5 Write property test for delta computation
    - **Property 10: Delta row equals dual-tower minus symmetric**
    - Generate random symmetric/dual-tower metric pairs, verify delta = dual_tower - symmetric for each metric
    - **Validates: Requirements 4.4**

  - [ ]* 7.6 Write property test for model fault isolation
    - **Property 11: Model fault isolation**
    - Generate random model lists with failure injection, verify non-failing models produce complete results and skipped_models contains exactly the failing ones
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.6**

- [x] 8. Final checkpoint — End-to-end verification
  - Ensure all tests pass, ask the user if questions arise.
  - Verify `run_benchmark.py` can be invoked with `python tests/embedding-benchmark/run_benchmark.py --models "BAAI/bge-m3"`
  - Verify `results.json` and `report.md` are generated correctly

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each task references specific requirements for traceability
- All 11 correctness properties from the design are covered as optional sub-tasks
- Property tests use `pytest` + `hypothesis` with minimum 100 iterations
- Test file: `tests/embedding-benchmark/test_benchmark.py`
- Reuses `tests/models.py` for `EmbeddingModelConfig` and `EMBEDDING_MODELS` — no modifications to existing files
- LLM for report: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` via same API base
