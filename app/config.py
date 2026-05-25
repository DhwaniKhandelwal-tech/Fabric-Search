"""
Configuration settings for Fabric Search API.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_DIR = DATA_DIR / "lancedb"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# ── LanceDB ────────────────────────────────────────────────────────────────
LANCEDB_URI = str(DB_DIR)
TABLE_NAME = "fabric_images"

# ── CLIP Model ────────────────────────────────────────────────────────────
# ViT-B/32 — good balance of speed and accuracy for fabric similarity
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBEDDING_DIM = 512          # ViT-B/32 output dimension

# ── GitHub source ─────────────────────────────────────────────────────────
GITHUB_API_URL = (
    "https://api.github.com/repos/recursivezero/foto/contents/assets/images/fresh"
)
GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/recursivezero/foto/main/assets/images/fresh"
)

# ── Search ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# ── API ────────────────────────────────────────────────────────────────────
API_TITLE = "Fabric Search API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Reverse image search for fabrics.

## Features
- **Image Search** — upload a fabric photo to find visually similar fabrics
- **Text Search** — search fabrics using natural language (e.g. "blue floral cotton")
- **Powered by** OpenCLIP embeddings + LanceDB vector store
"""
