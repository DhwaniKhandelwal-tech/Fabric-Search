from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP model %s on %s …", CLIP_MODEL_NAME, self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        self.model.eval()

    def _autocast(self):
        """autocast only on CUDA — CPU (macOS) doesn't support bfloat16."""
        if self.device == "cuda":
            return torch.amp.autocast("cuda")
        return torch.no_grad()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), self._autocast():
            features = self.model.encode_image(tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_image_bytes(self, data: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return self.embed_image(image)

    def embed_images_batch(self, images: List[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad(), self._autocast():
            features = self.model.encode_image(tensors)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad(), self._autocast():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
