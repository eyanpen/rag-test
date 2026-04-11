# Requirements Document

## Introduction

Build a retrieval-oriented benchmark tool that evaluates Embedding models under two encoding modes — symmetric (no prefix) and dual-tower / asymmetric (with query/document prefixes). The existing `tests/test_dual_tower_support.py` only compares a single query–document pair via cosine similarity, which cannot measure ranking quality. This feature constructs a multi-domain English evaluation dataset with one relevant document and multiple distractors per query, computes standard information-retrieval metrics (MRR, Recall@K, NDCG@K), compares models cross-wise, and uses an LLM to generate a structured Markdown analysis report. All code resides under `tests/embedding-benchmark/`.

## Glossary

- **Benchmark_Runner**: The main Python entry-point script (`tests/embedding-benchmark/run_benchmark.py`) that orchestrates dataset loading, embedding, metric computation, and report generation
- **Eval_Dataset**: A JSON file (`tests/embedding-benchmark/dataset.json`) containing an array of evaluation items, each with a query, one relevant document, and multiple distractor documents
- **Eval_Item**: A single evaluation unit consisting of a query string, a relevant_doc string, and a list of distractor_doc strings
- **Embedding_Client**: A helper module that calls the OpenAI-compatible `/embeddings` endpoint at `http://10.210.156.69:8633`, handles response sorting by `index`, and returns ordered embedding vectors
- **Retrieval_Evaluator**: A module that ranks candidate documents by cosine similarity to the query embedding and computes MRR, Recall@1, Recall@5, and NDCG@10
- **Symmetric_Mode**: Encoding mode where both query and document texts are embedded without any prefix
- **DualTower_Mode**: Encoding mode where the query text is prepended with the model's `query_prefix` and each document text is prepended with the model's `document_prefix`, as defined in `tests/models.py`
- **Report_Generator**: A module that sends benchmark results to the LLM API (`Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`) and writes the returned Markdown analysis to disk
- **EmbeddingModelConfig**: The dataclass defined in `tests/models.py` that holds model name, dimension, max tokens, display name, prefix strings, and dual-tower support flag
- **MRR**: Mean Reciprocal Rank — the average of 1/rank of the first relevant document across all queries
- **Recall_at_K**: The fraction of queries where the relevant document appears in the top-K ranked results
- **NDCG_at_K**: Normalized Discounted Cumulative Gain at cutoff K, measuring ranking quality with position-weighted relevance

## Requirements

### Requirement 1: English Evaluation Dataset Construction

**User Story:** As a developer, I want a curated English evaluation dataset with queries and candidate documents spanning multiple domains and difficulty levels, so that the benchmark produces meaningful and diverse retrieval assessments.

#### Acceptance Criteria

1. THE Eval_Dataset SHALL contain a minimum of 30 Eval_Items covering at least 5 distinct knowledge domains (e.g., science, history, technology, medicine, law)
2. Each Eval_Item SHALL contain exactly one query string, one relevant_doc string, and between 7 and 19 distractor_doc strings
3. THE relevant_doc in each Eval_Item SHALL be semantically related to the query and contain information that directly answers or addresses the query
4. Each distractor_doc SHALL be topically plausible but not a correct answer to the query, creating realistic retrieval difficulty
5. THE Eval_Dataset SHALL be stored as a JSON file at `tests/embedding-benchmark/dataset.json` with the schema: `[{"query": str, "relevant_doc": str, "distractor_docs": [str, ...], "domain": str, "difficulty": str}]`
6. THE Eval_Dataset SHALL include Eval_Items at three difficulty levels: "easy" (distractor topics clearly different), "medium" (distractors from same broad domain), and "hard" (distractors from same narrow sub-topic)
7. All query strings, relevant_doc strings, and distractor_doc strings SHALL be written in English

### Requirement 2: Embedding Client with Correct Response Ordering

**User Story:** As a developer, I want a reliable embedding client that calls the API and returns vectors in the correct input order, so that document embeddings are correctly matched to their source texts.

#### Acceptance Criteria

1. THE Embedding_Client SHALL call the `/embeddings` endpoint at `http://10.210.156.69:8633` using the OpenAI-compatible request format with fields `model` and `input`
2. THE Embedding_Client SHALL sort the returned `data` array by the `index` field before extracting embedding vectors, ensuring output order matches input order
3. THE Embedding_Client SHALL accept a `prefix` parameter and prepend the prefix to each input text before sending the request
4. THE Embedding_Client SHALL set a request timeout of 120 seconds per API call
5. IF the API returns an HTTP error status (4xx or 5xx), THEN THE Embedding_Client SHALL raise a descriptive exception containing the model name, status code, and response body
6. THE Embedding_Client SHALL support batching: when the input list exceeds 64 texts, the Embedding_Client SHALL split into batches of 64, call the API sequentially, and concatenate results in order

### Requirement 3: Dual-Mode Retrieval Evaluation

**User Story:** As a developer, I want each model evaluated in both symmetric and dual-tower modes using retrieval ranking metrics, so that I can determine whether prefix-based encoding improves retrieval quality.

#### Acceptance Criteria

1. FOR each EmbeddingModelConfig in the model list, THE Benchmark_Runner SHALL execute retrieval evaluation in Symmetric_Mode (no prefix for query or document)
2. FOR each EmbeddingModelConfig where `supports_dual_tower` is True, THE Benchmark_Runner SHALL also execute retrieval evaluation in DualTower_Mode (query_prefix for query, document_prefix for documents)
3. WHEN evaluating an Eval_Item, THE Retrieval_Evaluator SHALL embed the query and all candidate documents (1 relevant + N distractors), rank candidates by descending cosine similarity to the query, and record the rank position of the relevant document
4. THE Retrieval_Evaluator SHALL compute the following metrics across all Eval_Items: MRR, Recall_at_1, Recall_at_5, and NDCG_at_10
5. THE Retrieval_Evaluator SHALL compute metrics separately for each difficulty level ("easy", "medium", "hard") in addition to the overall aggregate
6. THE Benchmark_Runner SHALL record per-model, per-mode timing (total embedding time in seconds) for performance comparison

### Requirement 4: Cross-Model Comparison

**User Story:** As a developer, I want a side-by-side comparison of all models across both modes, so that I can identify which model and mode combination performs best for retrieval.

#### Acceptance Criteria

1. THE Benchmark_Runner SHALL produce a results summary containing, for each model–mode combination: model display name, mode (symmetric/dual-tower), MRR, Recall_at_1, Recall_at_5, NDCG_at_10, and total embedding time
2. THE Benchmark_Runner SHALL save the results summary as a JSON file at `tests/embedding-benchmark/results.json`
3. THE Benchmark_Runner SHALL rank model–mode combinations by MRR in descending order in the results summary
4. FOR each model that supports dual-tower, THE Benchmark_Runner SHALL compute a delta row showing the difference (DualTower minus Symmetric) for each metric

### Requirement 5: LLM-Generated Markdown Report

**User Story:** As a developer, I want an LLM to analyze the benchmark results and generate a structured Markdown report with insights, so that I get expert-level interpretation beyond raw numbers.

#### Acceptance Criteria

1. THE Report_Generator SHALL call the LLM API at `http://10.210.156.69:8633` using model `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` via the OpenAI-compatible `/chat/completions` endpoint
2. THE Report_Generator SHALL send the complete `results.json` content as context in the LLM prompt, along with instructions to produce a structured Markdown report
3. THE Report_Generator SHALL instruct the LLM to include the following sections in the report: Executive Summary, Methodology, Results Table (all models × all metrics), Dual-Tower vs Symmetric Analysis, Per-Difficulty Breakdown, Model Rankings, and Recommendations
4. THE Report_Generator SHALL save the LLM-generated report to `tests/embedding-benchmark/report.md`
5. IF the LLM API call fails, THEN THE Report_Generator SHALL fall back to generating a basic template report from the raw results data without LLM analysis
6. THE Report_Generator SHALL set a request timeout of 300 seconds for the LLM API call to accommodate long response generation

### Requirement 6: Error Handling and Graceful Degradation

**User Story:** As a developer, I want the benchmark to handle unavailable models and API errors gracefully, so that a single failure does not prevent the remaining models from being evaluated.

#### Acceptance Criteria

1. WHEN the Embedding_Client receives an HTTP 5xx response for a model, THE Benchmark_Runner SHALL log a warning, mark the model as "unavailable", and continue with the next model
2. WHEN the Embedding_Client receives an HTTP 4xx response for a model, THE Benchmark_Runner SHALL log an error with the response body, mark the model as "error", and continue with the next model
3. THE Benchmark_Runner SHALL perform a health-check embedding call (single short text) for each model before running the full evaluation, and skip models that fail the health check
4. THE Benchmark_Runner SHALL log all progress and errors to both stdout and a log file at `tests/embedding-benchmark/benchmark.log`
5. IF all models fail the health check, THEN THE Benchmark_Runner SHALL exit with a non-zero return code and a descriptive error message
6. THE results.json SHALL include a `skipped_models` array listing models that were skipped, each with the model name and the reason for skipping

### Requirement 7: Code Organization and Entry Point

**User Story:** As a developer, I want all benchmark code organized under `tests/embedding-benchmark/` with a clear entry point, so that I can run the benchmark with a single command.

#### Acceptance Criteria

1. THE Benchmark_Runner entry point SHALL be located at `tests/embedding-benchmark/run_benchmark.py` and executable via `python tests/embedding-benchmark/run_benchmark.py`
2. THE Benchmark_Runner SHALL import model configurations from `tests/models.py` using the existing `EMBEDDING_MODELS` list and `EmbeddingModelConfig` dataclass
3. THE Benchmark_Runner SHALL accept an optional `--models` command-line argument to specify a comma-separated list of model names to evaluate, defaulting to all models in `EMBEDDING_MODELS`
4. THE Benchmark_Runner SHALL accept an optional `--output-dir` command-line argument to specify the output directory, defaulting to `tests/embedding-benchmark/`
5. THE Benchmark_Runner SHALL print a progress summary after each model completes, showing the model name, mode, and computed metrics
6. WHEN the benchmark completes, THE Benchmark_Runner SHALL print the path to the generated report and results files
