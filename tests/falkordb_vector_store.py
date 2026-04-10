"""FalkorDB vector store implementation for Microsoft GraphRAG.

This module provides a custom VectorStore backend that stores and retrieves
vector embeddings using FalkorDB graph database. It is registered with the
GraphRAG vector store factory so that ``type: falkordb`` in settings.yaml
routes to this implementation.
"""

import json
import logging
from typing import Any

import numpy as np
from falkordb import FalkorDB

from graphrag_vectors import (
    VectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
    register_vector_store,
)

logger = logging.getLogger(__name__)


class FalkorDBVectorStore(VectorStore):
    """FalkorDB vector store implementation for GraphRAG."""

    def __init__(self, **kwargs: Any):
        # Extract FalkorDB-specific params before passing to super
        self.host = kwargs.pop("host", "10.210.156.69")
        self.port = kwargs.pop("port", 6379)
        self.graph_name = kwargs.pop("graph_name", "default")
        self.username = kwargs.pop("username", None)
        self.password = kwargs.pop("password", None)
        # Remove fields that VectorStore base doesn't expect
        kwargs.pop("type", None)
        kwargs.pop("db_uri", None)
        kwargs.pop("url", None)
        kwargs.pop("api_key", None)
        kwargs.pop("audience", None)
        kwargs.pop("connection_string", None)
        kwargs.pop("database_name", None)
        kwargs.pop("index_schema", None)
        super().__init__(**kwargs)
        self.db = None
        self.graph = None

    def connect(self) -> None:
        """Connect to FalkorDB and select graph."""
        self.db = FalkorDB(host=self.host, port=self.port)
        self.graph = self.db.select_graph(self.graph_name)
        logger.info(
            "Connected to FalkorDB %s:%d, graph=%s",
            self.host,
            self.port,
            self.graph_name,
        )

    def create_index(self) -> None:
        """Create vector index in FalkorDB (handled implicitly)."""
        pass

    def load_documents(self, documents: list[VectorStoreDocument]) -> None:
        """Load documents into the vector store."""
        for document in documents:
            self._insert_one(document)

    def insert(self, document: VectorStoreDocument) -> None:
        """Insert a single document with its vector into FalkorDB."""
        self._insert_one(document)

    def _insert_one(self, document: VectorStoreDocument) -> None:
        """Insert a single document with its vector into FalkorDB."""
        self._prepare_document(document)
        props: dict[str, Any] = {"id": str(document.id)}
        if document.vector is not None:
            props["vector"] = json.dumps(document.vector)
        # Store additional data fields
        for key, value in (document.data or {}).items():
            if isinstance(value, np.ndarray):
                props[key] = json.dumps(value.tolist())
            elif isinstance(value, (list, dict)):
                props[key] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)):
                props[key] = value
            else:
                props[key] = str(value)

        props_str = ", ".join(f"{k}: ${k}" for k in props)
        label = self.index_name
        query = f"CREATE (n:{label} {{{props_str}}})"
        self.graph.query(query, props)

    def similarity_search_by_vector(
        self,
        query_embedding: list[float],
        k: int = 10,
        select: list[str] | None = None,
        filters=None,
        include_vectors: bool = True,
    ) -> list[VectorStoreSearchResult]:
        """Vector similarity search using cosine similarity computed in Python."""
        label = self.index_name
        result = self.graph.query(
            f"MATCH (n:{label}) WHERE n.vector IS NOT NULL RETURN n"
        )

        candidates: list[VectorStoreSearchResult] = []
        query_arr = np.array(query_embedding)
        query_norm = np.linalg.norm(query_arr)

        for row in result.result_set:
            node = row[0]
            node_props = node.properties
            vec_raw = node_props.get("vector")
            if vec_raw is None:
                continue
            vec = json.loads(vec_raw) if isinstance(vec_raw, str) else list(vec_raw)

            # Cosine similarity
            doc_arr = np.array(vec)
            doc_norm = np.linalg.norm(doc_arr)
            if query_norm == 0 or doc_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(query_arr, doc_arr) / (query_norm * doc_norm))

            doc = VectorStoreDocument(
                id=node_props.get("id", ""),
                vector=vec if include_vectors else None,
                data={
                    k_: v
                    for k_, v in node_props.items()
                    if k_ not in ("id", "vector")
                },
            )
            candidates.append(VectorStoreSearchResult(document=doc, score=score))

        # Sort by score descending, return top k
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:k]

    def search_by_id(
        self,
        id: str,
        select: list[str] | None = None,
        include_vectors: bool = True,
    ) -> VectorStoreDocument:
        """Search for a document by id."""
        label = self.index_name
        result = self.graph.query(
            f"MATCH (n:{label} {{id: $id}}) RETURN n", {"id": id}
        )
        if not result.result_set:
            return VectorStoreDocument(id=id, vector=None)
        node = result.result_set[0][0]
        props = node.properties
        vec = None
        if include_vectors and "vector" in props:
            vec_raw = props["vector"]
            vec = json.loads(vec_raw) if isinstance(vec_raw, str) else list(vec_raw)
        return VectorStoreDocument(
            id=props.get("id", id),
            vector=vec,
            data={k: v for k, v in props.items() if k not in ("id", "vector")},
        )

    def count(self) -> int:
        """Return total number of documents in the store."""
        label = self.index_name
        result = self.graph.query(f"MATCH (n:{label}) RETURN count(n)")
        return result.result_set[0][0] if result.result_set else 0

    def remove(self, ids: list[str]) -> None:
        """Remove documents by id."""
        label = self.index_name
        for doc_id in ids:
            self.graph.query(
                f"MATCH (n:{label} {{id: $id}}) DELETE n", {"id": doc_id}
            )

    def update(self, document: VectorStoreDocument) -> None:
        """Update a document in the store."""
        self._prepare_update(document)
        label = self.index_name
        props: dict[str, Any] = {}
        if document.vector is not None:
            props["vector"] = json.dumps(document.vector)
        for key, value in (document.data or {}).items():
            if isinstance(value, np.ndarray):
                props[key] = json.dumps(value.tolist())
            elif isinstance(value, (list, dict)):
                props[key] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)):
                props[key] = value
            else:
                props[key] = str(value)

        if props:
            set_str = ", ".join(f"n.{k} = ${k}" for k in props)
            query = f"MATCH (n:{label} {{id: $id}}) SET {set_str}"
            props["id"] = str(document.id)
            self.graph.query(query, props)


# Register with GraphRAG factory
register_vector_store("falkordb", FalkorDBVectorStore)
