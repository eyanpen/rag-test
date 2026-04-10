"""Sync GraphRAG entities & relationships from Parquet into FalkorDB as a real graph."""
import logging
import os

import pandas as pd
from falkordb import FalkorDB

log = logging.getLogger(__name__)


def sync_graph_to_falkordb(
    output_dir: str,
    host: str,
    port: int,
    graph_name: str,
) -> None:
    """Read entities/relationships parquet and write nodes + edges into FalkorDB.

    Idempotent: uses MERGE to avoid duplicates on re-run.
    """
    entities_path = os.path.join(output_dir, "entities.parquet")
    relationships_path = os.path.join(output_dir, "relationships.parquet")
    if not os.path.isfile(entities_path) or not os.path.isfile(relationships_path):
        log.warning("Parquet files not found in %s, skipping graph sync", output_dir)
        return

    db = FalkorDB(host=host, port=port)
    graph = db.select_graph(graph_name)

    # ── Nodes ──
    entities_df = pd.read_parquet(entities_path)
    max_desc_len = entities_df["description"].str.len().max()
    log.info("Syncing %d entities to FalkorDB graph '%s' (max description: %d chars)", len(entities_df), graph_name, max_desc_len)
    for _, row in entities_df.iterrows():
        entity_type = str(row.get("type", "Entity")).replace(" ", "_")
        params = {
            "eid": str(row["id"]),
            "title": str(row.get("title", "")),
            "description": str(row.get("description", "")),
            "degree": int(row.get("degree", 0)),
        }
        graph.query(
            f"MERGE (n:{entity_type} {{id: $eid}}) "
            f"SET n.title = $title, n.description = $description, n.degree = $degree",
            params,
        )

    # ── Edges ──
    relationships_df = pd.read_parquet(relationships_path)
    log.info("Syncing %d relationships to FalkorDB graph '%s'", len(relationships_df), graph_name)
    for _, row in relationships_df.iterrows():
        params = {
            "src": str(row["source"]),
            "tgt": str(row["target"]),
            "rid": str(row["id"]),
            "desc": str(row.get("description", "")),
            "weight": float(row.get("weight", 1.0)),
        }
        graph.query(
            "MATCH (a {title: $src}), (b {title: $tgt}) "
            "MERGE (a)-[r:RELATED {id: $rid}]->(b) "
            "SET r.description = $desc, r.weight = $weight",
            params,
        )

    log.info(
        "Graph sync done: %s — Nodes: %d, Edges: %d",
        graph_name, len(entities_df), len(relationships_df),
    )
