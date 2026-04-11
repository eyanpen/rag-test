"""Root Cause 验证测试: embedding API 不支持 token IDs 输入格式

Root Cause: RayLLM/vLLM embedding endpoint 只接受文本字符串，不支持 token IDs 数组
证明方法: 分别用 token IDs 和文本字符串调用 /embeddings，验证前者 500、后者 200

运行方式:
  cd /home/eyanpen/sourceCode/rag-test
  source venv/bin/activate
  python -m pytest tests/test_rootcause_embedding_input_format.py -v
"""
import httpx
import pytest

API_BASE = "http://10.210.156.69:8633"
MODEL = "BAAI/bge-m3"


def _api_reachable() -> bool:
    try:
        httpx.get(f"{API_BASE}/health", timeout=5)
        return True
    except Exception:
        return False


skip_if_unreachable = pytest.mark.skipif(
    not _api_reachable(),
    reason=f"API {API_BASE} is not reachable",
)


@skip_if_unreachable
@pytest.mark.asyncio
async def test_token_ids_input_returns_500():
    """token IDs 格式输入 → 服务端返回 500（不支持此格式）"""
    payload = {
        "input": [[2, 7648, 7874, 4078, 315, 28049, 26211]],
        "model": MODEL,
        "encoding_format": "base64",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{API_BASE}/embeddings", json=payload)
    assert resp.status_code == 500, f"Expected 500, got {resp.status_code}"


@skip_if_unreachable
@pytest.mark.asyncio
async def test_text_string_input_returns_200():
    """文本字符串格式输入 → 服务端返回 200 并包含 embedding 数据"""
    payload = {
        "input": ["TUMORS: Tumors are abnormal tissue masses that result from uncontrolled cell growth."],
        "model": MODEL,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{API_BASE}/embeddings", json=payload)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) > 0
    assert "embedding" in data["data"][0]
