"""Data models, constants and utility functions for embedding benchmark."""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DualTowerMode(str, Enum):
    """Embedding encoding mode."""
    SYMMETRIC = "symmetric"     # query 和 document 使用相同编码
    ASYMMETRIC = "asymmetric"   # query 用 query_prefix，document 用 document_prefix


@dataclass
class EmbeddingModelConfig:
    name: str
    dim: int
    max_tokens: int
    display_name: str
    supports_dual_tower: bool = False
    query_prefix: str = ""
    document_prefix: str = ""


@dataclass
class BenchmarkConfig:
    api_base_url: str = "http://10.210.156.69:8633"
    llm_model: str = "deepseek-ai/DeepSeek-V3.2"
    falkordb_host: str = "10.210.156.69"
    falkordb_port: int = 6379
    sample_size: int = 5
    datasets: List[str] = field(default_factory=lambda: ["medical"])
    models: List["EmbeddingModelConfig"] = field(default_factory=lambda: EMBEDDING_MODELS)
    output_dir: str = "tests/benchmark_results"
    data_root: str = "GraphRAG-Benchmark/Datasets"
    graphrag_root: str = "/home/eyanpen/sourceCode/rnd-ai-engine-features/graphrag"


@dataclass
class DatasetPhaseResult:
    """数据集 Phase 1-9 共享阶段结果"""
    dataset_name: str
    phase_1_9_time_seconds: float
    output_dir: str
    error: Optional[str] = None


@dataclass
class PredictionItem:
    id: str
    question: str
    source: str
    context: List[str]
    generated_answer: str
    ground_truth: str
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    model_name: str
    dataset_name: str
    rouge_l: float
    answer_correctness: float
    eval_time_seconds: float


@dataclass
class ModelResult:
    model: EmbeddingModelConfig
    dataset_name: str
    predictions: List[PredictionItem] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    embedding_time_seconds: float = 0.0
    query_time_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    config: BenchmarkConfig
    dataset_phases: List[DatasetPhaseResult] = field(default_factory=list)
    model_results: List[ModelResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_time_seconds: float = 0.0


EMBEDDING_MODELS = [
    # BGE-M3: vLLM API 下三个变体返回完全相同的向量，但支持 query:/passage: prefix 双塔 (cos(q,d)=0.970)
    EmbeddingModelConfig("BAAI/bge-m3", 1024, 8192, "BGE-M3 (default)",
        supports_dual_tower=True,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    EmbeddingModelConfig("BAAI/bge-m3/heavy", 1024, 8192, "BGE-M3 (heavy)",
        supports_dual_tower=True,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    EmbeddingModelConfig("BAAI/bge-m3/interactive", 1024, 8192, "BGE-M3 (interactive)",
        supports_dual_tower=True,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    # E5-Mistral: query prefix 效果显著 (cos(bare,q)=0.787)
    EmbeddingModelConfig(
        "intfloat/e5-mistral-7b-instruct", 4096, 4096, "E5-Mistral-7B",
        supports_dual_tower=True,
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        document_prefix="",
    ),
    # mE5-Large: query/doc prefix 都有效 (cos(q,d)=0.980)
    EmbeddingModelConfig("intfloat/multilingual-e5-large-instruct", 1024, 512, "mE5-Large",
        supports_dual_tower=True,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    # Nomic: query/doc prefix 效果最明显 (cos(q,d)=0.934)
    EmbeddingModelConfig("nomic-ai/nomic-embed-text-v1.5", 768, 8192, "Nomic-v1.5",
        supports_dual_tower=True,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    ),
    # Qwen3-Emb-8B: query prefix 效果显著 (cos(bare,q)=0.804)
    EmbeddingModelConfig("Qwen/Qwen3-Embedding-8B", 4096, 8192, "Qwen3-Emb-8B",
        supports_dual_tower=True,
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        document_prefix="",
    ),
    # Qwen3-Emb-8B-Alt: API 返回 500，当前不可用
    EmbeddingModelConfig("Qwen/Qwen3-Embedding-8B-Alt", 4096, 32768, "Qwen3-Emb-8B-Alt",
        supports_dual_tower=True,
        query_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        document_prefix="",
    ),
]


def sanitize_name(model_name: str) -> str:
    """Replace '/' with '-' and convert to lowercase. Used for FalkorDB graph naming."""
    return model_name.replace("/", "-").lower()


def make_graph_name(model_name: str, dataset_name: str, dual_tower: bool = False) -> str:
    """Generate FalkorDB graph name: dataset + '_' + sanitize(model) [+ '_dualtower']."""
    base = sanitize_name(model_name)
    if dual_tower:
        base += "_dualtower"
    return dataset_name + "_" + base
