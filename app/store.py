from __future__ import annotations

import logging
from typing import Any, Dict, List

import lancedb
import numpy as np
import pyarrow as pa

from app.config import (
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    LANCEDB_URI,
    MAX_TOP_K,
    TABLE_NAME,
)

logger = logging.getLogger(__name__)

_SCHEMA = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    pa.field("id", pa.string()),
    pa.field("filename", pa.string()),
    pa.field("url", pa.string()),
    pa.field("local_path", pa.string()),
    pa.field("tags", pa.string()),
    pa.field("created_at", pa.string()),
])


def _get_db():
    return lancedb.connect(LANCEDB_URI)


def get_or_create_table():
    db = _get_db()
    if TABLE_NAME not in db.table_names():
        logger.info("Creating LanceDB table '%s'", TABLE_NAME)
        return db.create_table(TABLE_NAME, schema=_SCHEMA)
    return db.open_table(TABLE_NAME)


def drop_table():
    db = _get_db()
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
        logger.info("Dropped table '%s'", TABLE_NAME)


def upsert_records(records: List[Dict[str, Any]]):
    if not records:
        return
    for r in records:
        if r.get("tags") is None:
            r["tags"] = ""
    table = get_or_create_table()
    table.add(records)
    logger.info("Inserted %d records", len(records))


def build_vector_index():
    table = get_or_create_table()
    row_count = table.count_rows()
    if row_count < 256:
        logger.info("Skipping ANN index (%d rows)", row_count)
        return
    logger.info("Building index on %d rows", row_count)
    table.create_index(metric="cosine", vector_column_name="vector", num_sub_vectors=64)
    logger.info("Index built.")


def vector_search(query_vector: np.ndarray, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    top_k = min(top_k, MAX_TOP_K)
    table = get_or_create_table()
    results = (
        table.search(query_vector)
        .metric("cosine")
        .limit(top_k)
        .to_list()
    )
    cleaned = []
    for row in results:
        row.pop("vector", None)
        distance = float(row.pop("_distance", row.pop("_relevance_score", 0.0)))
        row["score"] = round(max(0.0, 1.0 - distance), 4)
        row["tags"] = row.get("tags") or ""
        cleaned.append(row)
    return cleaned


def get_all_ids() -> List[str]:
    table = get_or_create_table()
    if table.count_rows() == 0:
        return []
    return [r["id"] for r in table.search().limit(100_000).to_list()]


def count_rows() -> int:
    return get_or_create_table().count_rows()