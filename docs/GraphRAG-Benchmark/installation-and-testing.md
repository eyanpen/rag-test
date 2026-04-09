# GraphRAG-Benchmark 安装与测试文档

> 版本：1.0 | 日期：2026-04-09

---

## 1. 环境要求

| 项目 | 要求 | 说明 |
|------|------|------|
| Python | 3.10+ | 推荐 3.10，部分框架对 3.11+ 兼容性未验证 |
| 包管理 | Conda（强烈推荐） | 各框架依赖可能冲突，建议独立环境 |
| GPU | 可选 | Embedding 模型加载和推理可利用 GPU 加速 |
| 操作系统 | Linux / macOS / Windows | Linux 推荐 |
| 磁盘空间 | ≥ 10GB | 数据集 + Embedding 模型 + 索引文件 |
| 内存 | ≥ 16GB | 大语料索引构建时内存消耗较高 |

---

## 2. 基础安装

### 2.1 克隆项目

```bash
git clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git
cd GraphRAG-Benchmark
```

### 2.2 安装基础依赖

```bash
pip install -r requirements.txt
```

**依赖清单（requirements.txt）：**

| 包名 | 版本 | 用途 |
|------|------|------|
| `datasets` | 3.3.2 | HuggingFace 数据集加载（Parquet） |
| `Evaluation` | 0.0.2 | 评估基础库 |
| `langchain` | 0.3.26 | LLM 调用框架 |
| `langchain_core` | 0.3.69 | LangChain 核心组件 |
| `langchain_openai` | 0.3.28 | OpenAI 兼容 API 集成 |
| `langchain_ollama` | (latest) | Ollama LLM/Embedding 集成 |
| `numpy` | (latest) | 数值计算 |
| `pydantic` | 2.11.7 | 数据模型验证 |
| `ragas` | 0.2.15 | RAG 评估指标库 |
| `rouge_score` | 0.1.2 | ROUGE 文本匹配评分 |
| `json5` | (latest) | 宽松 JSON 解析 |
| `json_repair` | (latest) | JSON 自动修复 |

> **注意：** 以上仅为评估模块的基础依赖。各 GraphRAG 框架有各自的依赖，需在独立环境中安装。

### 2.3 下载 Embedding 模型

评估和推理都需要 Embedding 模型，推荐使用 `BAAI/bge-large-en-v1.5`：

```bash
# 方式一：通过 HuggingFace 自动下载（首次运行时）
# 评估脚本中指定 --embedding_model BAAI/bge-large-en-v1.5 即可

# 方式二：手动下载到本地（推荐，避免网络问题）
git lfs install
git clone https://huggingface.co/BAAI/bge-large-en-v1.5 /path/to/bge-large-en-v1.5
```

---

## 3. 各框架独立安装

> ⚠️ **强烈建议每个框架使用独立的 Conda 环境**，避免依赖冲突。

### 3.1 LightRAG（v1.2.5）

```bash
# 创建独立环境
conda create -n lightrag python=3.10 -y
conda activate lightrag

# 安装 LightRAG
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
git checkout v1.2.5  # 确保版本
pip install -e .

# 安装评估基础依赖
cd ../GraphRAG-Benchmark
pip install -r requirements.txt
```

**必须的源码修改：**

LightRAG 默认不返回检索上下文，需修改两处：

**修改 1：`lightrag/operate.py`**

```python
# 原始代码
async def kg_query(...) -> str | AsyncIterator[str]:
    return response

# 修改为
async def kg_query(...) -> tuple[str, str] | tuple[AsyncIterator[str], str]:
    return response, context
```

**修改 2：`lightrag/lightrag.py`**

```python
# 修改 aquery 方法
async def aquery(...):
    ...
    if param.mode in ["local", "global", "hybrid"]:
        response, context = await kg_query(...)
    ...
    return response, context
```

### 3.2 fast-graphrag

```bash
conda create -n fast-graphrag python=3.10 -y
conda activate fast-graphrag

# 安装 fast-graphrag
git clone https://github.com/circlemind-ai/fast-graphrag.git
cd fast-graphrag
pip install -e .

# 安装评估基础依赖
cd ../GraphRAG-Benchmark
pip install -r requirements.txt
```

**必须的源码修改：**

fast-graphrag 不支持 HuggingFace Embedding，需手动添加：

**新增文件：`fast_graphrag/_llm/_hf.py`**

创建 `HuggingFaceEmbeddingService` 类，实现基于 HuggingFace transformers 的 Embedding 服务。该类继承 `BaseEmbeddingService`，支持：
- 批量编码（`max_elements_per_request=32`）
- GPU/MPS/CPU 自动设备选择
- 异步并发限流
- 指数退避重试（3 次）

> 完整代码见 `Examples/README.md` 中的 fast-graphrag 章节。

**修改文件：`fast_graphrag/_llm/__init__.py`**

```python
__all__ = [
    ...
    "HuggingFaceEmbeddingService",
]
...
from ._hf import HuggingFaceEmbeddingService
```

### 3.3 HippoRAG2（v1.0.0）

```bash
conda create -n hipporag2 python=3.10 -y
conda activate hipporag2

# 安装 HippoRAG2
git clone https://github.com/OSU-NLP-Group/HippoRAG.git
cd HippoRAG
git checkout v1.0.0
pip install -e .

# 安装评估基础依赖
cd ../GraphRAG-Benchmark
pip install -r requirements.txt
```

**必须的源码修改：**

HippoRAG2 不支持 BGE Embedding 模型，需手动添加：

**新增文件：`hipporag/embedding_model/BGE.py`**

创建 `BGEEmbeddingModel` 类，实现基于 BGE 模型的 Embedding 服务。支持：
- Mean pooling 策略
- 可选的 L2 归一化
- 多 GPU 自动分配（`device_map="auto"`）
- 查询和语料分别编码（带 instruction 前缀）

> 完整代码见 `Examples/README.md` 中的 HippoRAG2 章节。

**修改文件：`hipporag/embedding_model/__init__.py`**

```python
from .BGE import BGEEmbeddingModel

def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    ...
    elif "bge" in embedding_model_name.lower():
        return BGEEmbeddingModel
    ...
```

### 3.4 DIGIMON

```bash
conda create -n digimon python=3.10 -y
conda activate digimon

# 安装 DIGIMON
git clone https://github.com/JayLZhou/GraphRAG.git
cd GraphRAG
pip install -e .
```

**使用方式：**

1. 将 `GraphRAG-Benchmark/Examples/run_digimon.py` 复制到 DIGIMON 项目根目录
2. 按 DIGIMON 文档修改 `Option/Method/` 下的 YAML 配置文件
3. 从 DIGIMON 项目目录运行

---

## 4. 运行推理测试

### 4.1 LightRAG

```bash
conda activate lightrag
cd GraphRAG-Benchmark

# API 模式
export LLM_API_KEY=<your-api-key>

python Examples/run_lightrag.py \
  --subset medical \
  --mode API \
  --base_dir ./Examples/lightrag_workspace \
  --model_name gpt-4o-mini \
  --embed_model BAAI/bge-large-en-v1.5 \
  --retrieve_topk 5 \
  --llm_base_url https://api.openai.com/v1

# Ollama 模式
python Examples/run_lightrag.py \
  --subset medical \
  --mode ollama \
  --base_dir ./Examples/lightrag_workspace \
  --model_name qwen2.5:14b \
  --embed_model BAAI/bge-large-en-v1.5 \
  --retrieve_topk 5 \
  --llm_base_url http://localhost:11434

# 快速测试（仅采样少量问题）
python Examples/run_lightrag.py \
  --subset medical \
  --mode API \
  --base_dir ./Examples/lightrag_workspace \
  --model_name gpt-4o-mini \
  --embed_model BAAI/bge-large-en-v1.5 \
  --sample 10 \
  --llm_base_url https://api.openai.com/v1
```

**输出路径：** `./results/lightrag/<corpus_name>/predictions_<corpus_name>.json`

### 4.2 fast-graphrag

```bash
conda activate fast-graphrag
cd GraphRAG-Benchmark

export LLM_API_KEY=<your-api-key>

python Examples/run_fast-graphrag.py \
  --subset medical \
  --mode API \
  --base_dir ./Examples/fast-graphrag_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path /path/to/bge-large-en-v1.5 \
  --llm_base_url https://api.openai.com/v1
```

**输出路径：** `./results/fast-graphrag/<corpus_name>/predictions_<corpus_name>.json`

### 4.3 HippoRAG2

```bash
conda activate hipporag2
cd GraphRAG-Benchmark

export OPENAI_API_KEY=<your-api-key>

python Examples/run_hipporag2.py \
  --subset medical \
  --mode API \
  --base_dir ./Examples/hipporag2_workspace \
  --model_name gpt-4o-mini \
  --embed_model_path /path/to/bge-large-en-v1.5 \
  --llm_base_url https://api.openai.com/v1
```

> ⚠️ HippoRAG2 使用 `OPENAI_API_KEY` 环境变量（非 `LLM_API_KEY`）。

**输出路径：** `./results/hipporag2/<corpus_name>/predictions_<corpus_name>.json`

### 4.4 DIGIMON

```bash
conda activate digimon
cd DIGIMON_PROJECT_DIR  # DIGIMON 项目目录

python run_digimon.py \
  --subset novel \
  --option ./Option/Method/HippoRAG.yaml \
  --output_dir ./results/test
```

**输出路径：** `./results/test/<corpus_name>_predictions.json`

---

## 5. 运行评估测试

### 5.1 Generation 评估（生成质量）

```bash
cd GraphRAG-Benchmark

export LLM_API_KEY=<your-api-key>

# 基础评估
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/generation_results.json

# 详细输出（包含每条问题的评分明细）
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/generation_results_detailed.json \
  --detailed_output

# 采样评估（每种问题类型仅评估 N 条）
python -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/generation_results.json \
  --num_samples 20
```

### 5.2 Retrieval 评估（检索质量）

```bash
export LLM_API_KEY=<your-api-key>

python -m Evaluation.retrieval_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model BAAI/bge-large-en-v1.5 \
  --data_file ./results/lightrag.json \
  --output_file ./results/retrieval_results.json
```

### 5.3 Indexing 评估（图构建质量）

```bash
# LightRAG
python -m Evaluation.indexing_eval \
  --framework lightrag \
  --base_path ./Examples/lightrag_workspace \
  --folder_name graph_store \
  --output ./results/indexing_lightrag.txt

# fast-graphrag
python -m Evaluation.indexing_eval \
  --framework fast_graphrag \
  --base_path ./Examples/fast-graphrag_workspace \
  --output ./results/indexing_fast_graphrag.txt

# HippoRAG2（需要指定 folder_name）
python -m Evaluation.indexing_eval \
  --framework hipporag2 \
  --base_path ./Examples/hipporag2_workspace \
  --folder_name graph_store \
  --output ./results/indexing_hipporag2.txt

# Microsoft GraphRAG
python -m Evaluation.indexing_eval \
  --framework microsoft_graphrag \
  --base_path ./graphrag_workspace \
  --output ./results/indexing_ms_graphrag.txt
```

> Indexing 评估不需要 LLM，纯图论计算，速度很快。

### 5.4 Ollama 模式评估

所有评估命令将 `--mode` 改为 `ollama`，`--base_url` 改为 Ollama 地址：

```bash
# 确保 Ollama 服务已启动
ollama serve

python -m Evaluation.generation_eval \
  --mode ollama \
  --model qwen2.5:14b \
  --base_url http://localhost:11434 \
  --embedding_model qwen2.5:14b \
  --data_file ./results/lightrag.json \
  --output_file ./results/generation_results_ollama.json
```

---

## 6. 验证安装清单

### 6.1 基础环境检查

```bash
# 检查 Python 版本
python --version  # 应为 3.10+

# 检查核心依赖
python -c "import datasets; print(f'datasets: {datasets.__version__}')"
python -c "import langchain; print(f'langchain: {langchain.__version__}')"
python -c "import ragas; print(f'ragas: {ragas.__version__}')"
python -c "import rouge_score; print('rouge_score: OK')"
python -c "import json5; print('json5: OK')"
python -c "import json_repair; print('json_repair: OK')"
python -c "import igraph; print(f'igraph: {igraph.__version__}')"
```

### 6.2 数据集检查

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('parquet', data_files='Datasets/Corpus/medical.parquet', split='train')
print(f'Medical corpus: {len(ds)} documents')
print(f'Columns: {ds.column_names}')
print(f'First corpus_name: {ds[0][\"corpus_name\"]}')
"
```

### 6.3 Embedding 模型检查

```bash
python -c "
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
emb = HuggingFaceBgeEmbeddings(model_name='BAAI/bge-large-en-v1.5')
result = emb.embed_query('test')
print(f'Embedding dim: {len(result)}')  # 应为 1024
print('Embedding model: OK')
"
```

### 6.4 LLM 连接检查

```bash
# API 模式
export LLM_API_KEY=<your-api-key>
python -c "
import os
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
llm = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(os.getenv('LLM_API_KEY')))
resp = llm.invoke('Say hello')
print(f'LLM response: {resp.content}')
print('API mode: OK')
"

# Ollama 模式
python -c "
import asyncio
from Evaluation.llm.ollama_client import OllamaClient
async def test():
    client = OllamaClient('http://localhost:11434')
    resp = await client.ainvoke('Say hello', model='qwen2.5:14b')
    print(f'Ollama response: {resp.content}')
    await client.close()
asyncio.run(test())
print('Ollama mode: OK')
"
```

---

## 7. 常见问题排查

### 7.1 API Key 未设置

```
ValueError: LLM_API_KEY environment variable is not set
```

**解决：** 设置环境变量：
```bash
export LLM_API_KEY=<your-api-key>        # LightRAG / fast-graphrag / 评估
export OPENAI_API_KEY=<your-api-key>     # HippoRAG2
```

### 7.2 Embedding 模型路径错误

```
OSError: Can't load tokenizer for '/home/xzs/data/model/bge-large-en-v1.5'
```

**解决：** `--embed_model_path` 需指向本地已下载的模型目录，或使用 HuggingFace Hub 名称 `BAAI/bge-large-en-v1.5`。

### 7.3 Ollama 服务未启动

```
ValueError: Failed to connect to Ollama service: Cannot connect to host localhost:11434
```

**解决：**
```bash
ollama serve                    # 启动服务
ollama pull qwen2.5:14b        # 拉取模型
```

### 7.4 内存不足（OOM）

索引构建大语料时可能 OOM。

**解决：**
- 使用 `--sample N` 参数限制处理的问题数量
- 减小 `chunk_token_size`（HippoRAG2 默认 256）
- 使用更小的 Embedding 模型

### 7.5 依赖冲突

不同框架的依赖版本可能冲突。

**解决：** 严格使用独立 Conda 环境，每个框架一个环境。

### 7.6 igraph 安装失败

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev libigraph-dev
pip install python-igraph

# macOS
brew install igraph
pip install python-igraph
```

### 7.7 评估输入文件格式错误

评估脚本要求输入 JSON 文件为列表格式，每个元素包含必需字段：

```json
[
  {
    "id": "...",
    "question": "...",
    "question_type": "Fact Retrieval",
    "context": ["..."],
    "evidence": ["..."],
    "generated_answer": "...",
    "ground_truth": "..."
  }
]
```

如果缺少字段或格式不对，评估会报错或跳过。
