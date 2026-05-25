# Fabric Search API

Reverse image search for fabrics — upload a fabric photo and find visually similar ones.  
Replicates the search feature at [lab.threadzip.com/find](https://lab.threadzip.com/find).

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | [OpenCLIP](https://github.com/mlfoundations/open_clip) `ViT-B/32` |
| Vector DB | [LanceDB](https://lancedb.github.io/lancedb/) |
| API | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Images | [recursivezero/foto](https://github.com/recursivezero/foto/tree/main/assets/images/fresh) |

---

## Project Structure

```
fabric-search/
├── app/
│   ├── __init__.py
│   ├── config.py        ← paths, constants, model name
│   ├── embeddings.py    ← OpenCLIP wrapper (image + text embeddings)
│   ├── store.py         ← LanceDB schema, read/write helpers
│   ├── schemas.py       ← Pydantic request/response models
│   └── main.py          ← FastAPI app & all endpoints
├── scripts/
│   └── ingest.py        ← download images → embed → store in LanceDB
├── data/                ← auto-created at runtime
│   ├── images/          ← downloaded fabric images cached here
│   └── lancedb/         ← LanceDB files
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd fabric-search

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Ingest Images

This downloads every image from the GitHub repo, computes CLIP embeddings, and stores them in LanceDB.

```bash
python -m scripts.ingest
```

Options:
```bash
python -m scripts.ingest --reset   # wipe DB and re-index from scratch
```

You will see progress like:
```
09:01:12  INFO     Loading CLIP model ViT-B-32 on cpu …
09:01:18  INFO     Found 120 image(s) in repo.
Indexing: 100%|████████████████| 120/120 [02:45<00:00,  0.72img/s]
09:04:03  INFO     Ingest complete. Added 120 image(s). Total: 120
09:04:03  INFO     Building vector index …
09:04:04  INFO     Done! ✓
```

### 3. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

### `POST /api/v1/search/image`
Upload a fabric image and find the most similar ones.

```bash
curl -X POST http://localhost:8000/api/v1/search/image \
  -F "file=@my_fabric.jpg" \
  -F "top_k=5"
```

**Response:**
```json
{
  "query_type": "image",
  "total": 5,
  "results": [
    {
      "id": "plaid_cotton_01",
      "filename": "plaid_cotton_01.jpg",
      "url": "https://raw.githubusercontent.com/...",
      "tags": "plaid, cotton",
      "score": 0.9712
    },
    ...
  ]
}
```

---

### `POST /api/v1/search/text`
Search fabrics by natural language description.

```bash
curl -X POST http://localhost:8000/api/v1/search/text \
  -F "query=red floral print on white background" \
  -F "top_k=8"
```

---

### `GET /api/v1/images/{id}`
Get metadata for a specific indexed image.

```bash
curl http://localhost:8000/api/v1/images/plaid_cotton_01
```

---

### `GET /api/v1/health`
```bash
curl http://localhost:8000/api/v1/health
# → {"status":"ok","indexed_images":120,"model":"ViT-B-32"}
```

### `GET /api/v1/stats`
```bash
curl http://localhost:8000/api/v1/stats
```

---

## How It Works

```
               ┌──────────────────────────────────────┐
  Query Image  │  1. Load image                        │
  (or text)    │  2. CLIP encode → float32[512] vector │
               │  3. LanceDB ANN cosine search         │
               │  4. Return top-K + metadata           │
               └──────────────────────────────────────┘

               ┌──────────────────────────────────────┐
  Ingest       │  1. GitHub API → list image URLs      │
  (one-time)   │  2. Download images (cached locally)  │
               │  3. CLIP batch encode                 │
               │  4. Store vectors + metadata in       │
               │     LanceDB IVF-PQ index              │
               └──────────────────────────────────────┘
```

**OpenCLIP** understands both images and text in the same embedding space — so text queries like `"striped linen fabric"` work alongside image queries.

**LanceDB** is a local, serverless vector database — no Docker, no cloud account needed. Data lives in `data/lancedb/`.

---

## Python Usage (programmatic)

```python
from app.embeddings import get_embedding_model
from app.store import vector_search
from PIL import Image

model = get_embedding_model()

# Search by image
img = Image.open("my_fabric.jpg").convert("RGB")
vec = model.embed_image(img)
results = vector_search(vec, top_k=5)

# Search by text
vec = model.embed_text("dark blue denim with diagonal weave")
results = vector_search(vec, top_k=5)

for r in results:
    print(r["filename"], r["score"])
```

---

## Notes

- First run of `ingest.py` takes ~3–5 minutes (downloads + embeds ~120 images).
- Subsequent runs skip already-indexed images.
- On CPU, each embedding takes ~10 ms (ViT-B/32). GPU is ~10× faster.
- LanceDB files are stored in `data/lancedb/` and persist across restarts.
