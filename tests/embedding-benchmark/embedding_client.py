"""Embedding API client with prefix support, batching, and response ordering."""

import math
from typing import List

import requests


class EmbeddingAPIError(Exception):
    """Raised when the embedding API returns an HTTP error."""

    def __init__(self, model: str, status_code: int, response_body: str):
        self.model = model
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(
            f"Embedding API error for model '{model}': "
            f"HTTP {status_code} - {response_body}"
        )


class EmbeddingClient:
    """OpenAI-compatible embedding client with prefix support and batching."""

    BATCH_SIZE = 64

    def __init__(self, api_base: str, timeout: int = 120):
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def get_embeddings(
        self,
        model: str,
        texts: List[str],
        prefix: str = "",
    ) -> List[List[float]]:
        """Embed texts with optional prefix prepended to each.

        - Prepends ``prefix`` to each text before sending
        - Splits into batches of 64 if len(texts) > 64
        - Sorts response ``data`` array by ``index`` field
        - Returns embedding vectors in input order
        - Raises EmbeddingAPIError on 4xx/5xx with model name, status, body
        """
        prepared = [prefix + t for t in texts]

        if len(prepared) > self.BATCH_SIZE:
            num_batches = math.ceil(len(prepared) / self.BATCH_SIZE)
            all_embeddings: List[List[float]] = []
            for i in range(num_batches):
                start = i * self.BATCH_SIZE
                end = start + self.BATCH_SIZE
                batch = prepared[start:end]
                all_embeddings.extend(self._call_api(model, batch))
            return all_embeddings
        else:
            return self._call_api(model, prepared)

    def _call_api(
        self, model: str, texts: List[str]
    ) -> List[List[float]]:
        """Send a single embedding request and return vectors sorted by index."""
        url = f"{self.api_base}/embeddings"
        payload = {"model": model, "input": texts}

        resp = requests.post(url, json=payload, timeout=self.timeout)

        if resp.status_code >= 400:
            raise EmbeddingAPIError(model, resp.status_code, resp.text)

        data = resp.json()["data"]
        sorted_data = sorted(data, key=lambda item: item["index"])
        return [item["embedding"] for item in sorted_data]
