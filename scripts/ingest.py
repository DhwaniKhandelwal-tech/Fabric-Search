"""
Ingestion pipeline
==================
Downloads fabric images from the GitHub repo, computes CLIP embeddings,
and stores everything in LanceDB.

Run once (or re-run to add new images):
    python -m scripts.ingest
    python -m scripts.ingest --reset   # drop table and re-index everything
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image
from tqdm import tqdm

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    GITHUB_API_URL,
    GITHUB_RAW_BASE,
    IMAGES_DIR,
)
from app.embeddings import get_embedding_model
from app.store import (
    build_vector_index,
    count_rows,
    drop_table,
    get_all_ids,
    upsert_records,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = 16          # images per embedding batch
REQUEST_TIMEOUT = 20     # seconds


# ── Helpers ────────────────────────────────────────────────────────────────

def _infer_tags(filename: str) -> str:
    """
    Derive simple tags from the filename.
    e.g. "red_floral_cotton_01.jpg"  →  "red, floral, cotton"
    """
    stem = Path(filename).stem.lower()
    # Split on underscores, hyphens, digits
    parts = re.split(r"[_\-\d]+", stem)
    tags = [p.strip() for p in parts if len(p) > 2]
    return ", ".join(dict.fromkeys(tags))   # preserve order, deduplicate


def _fetch_github_file_list() -> List[Dict[str, str]]:
    """
    Use the GitHub Contents API to list files in the fresh/ folder.
    Returns list of dicts with 'name' and 'download_url'.
    """
    logger.info("Fetching file list from GitHub …")
    headers = {"Accept": "application/vnd.github+json"}

    try:
        resp = requests.get(GITHUB_API_URL, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        entries = resp.json()
    except requests.RequestException as exc:
        logger.error("GitHub API request failed: %s", exc)
        logger.info("Falling back to raw URL scanning (not available).")
        return []

    files = [
        {"name": e["name"], "download_url": e["download_url"]}
        for e in entries
        if isinstance(e, dict)
        and e.get("type") == "file"
        and Path(e["name"]).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    logger.info("Found %d image(s) in repo.", len(files))
    return files


def _download_image(
    url: str, dest: Path, retries: int = 3
) -> Optional[Image.Image]:
    """Download an image to disk (if not already present) and return it."""
    if dest.exists():
        try:
            return Image.open(dest).convert("RGB")
        except Exception:
            dest.unlink(missing_ok=True)

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            resp.raise_for_status()
            raw = resp.content
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            dest.write_bytes(raw)
            return img
        except Exception as exc:
            logger.warning("Attempt %d/%d failed for %s: %s", attempt + 1, retries, url, exc)
            time.sleep(1)
    return None


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_ingest(reset: bool = False) -> None:
    if reset:
        logger.info("--reset flag set: dropping existing table.")
        drop_table()

    # Load embedding model
    model = get_embedding_model()

    # Get list of images from GitHub
    file_list = _fetch_github_file_list()
    if not file_list:
        logger.error("No images found. Check your internet connection or GitHub token.")
        sys.exit(1)

    # Determine which IDs are already indexed
    existing_ids = set(get_all_ids())
    logger.info("Already indexed: %d image(s).", len(existing_ids))

    to_process = [
        f for f in file_list
        if Path(f["name"]).stem not in existing_ids
    ]
    logger.info("To index: %d image(s).", len(to_process))

    if not to_process:
        logger.info("Nothing new to index. Total in DB: %d", count_rows())
        return

    # Process in batches
    total_added = 0
    batch_imgs: List[Image.Image] = []
    batch_meta: List[Dict[str, Any]] = []

    def _flush_batch() -> None:
        nonlocal total_added
        if not batch_imgs:
            return
        vectors = model.embed_images_batch(batch_imgs)
        records = []
        for i, meta in enumerate(batch_meta):
            records.append({**meta, "vector": vectors[i].tolist()})
        upsert_records(records)
        total_added += len(records)
        batch_imgs.clear()
        batch_meta.clear()

    for file_info in tqdm(to_process, desc="Indexing", unit="img"):
        filename = file_info["name"]
        download_url = file_info["download_url"]
        stem = Path(filename).stem
        local_path = IMAGES_DIR / filename

        img = _download_image(download_url, local_path)
        if img is None:
            logger.warning("Skipping %s (download failed).", filename)
            continue

        batch_imgs.append(img)
        batch_meta.append(
            {
                "id": stem,
                "filename": filename,
                "url": download_url,
                "local_path": str(local_path),
                "tags": _infer_tags(filename),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        if len(batch_imgs) >= BATCH_SIZE:
            _flush_batch()

    _flush_batch()  # flush remainder

    logger.info("Ingest complete. Added %d image(s). Total: %d", total_added, count_rows())

    # Build ANN index for fast search
    logger.info("Building vector index …")
    build_vector_index()
    logger.info("Done! ✓")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest fabric images into LanceDB.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop existing table and re-index from scratch.",
    )
    args = parser.parse_args()
    run_ingest(reset=args.reset)
