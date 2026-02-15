"""Lightweight classifier for determining modality and quality."""

from __future__ import annotations

import logging
from typing import Dict

from ..config import AppConfig

LOGGER = logging.getLogger("scanner.classifier")


class ModalityClassifier:
    def __init__(self, config: AppConfig):
        self.config = config

    def classify(self, modality: str, entropy: float, attributes: Dict) -> Dict:
        score = entropy if modality != "text" else min(entropy * 1.5, 1.0)
        is_valid = score >= self._threshold(modality)
        LOGGER.debug("Classified %s with score %.2f", modality, score)
        return {"score": score, "valid": is_valid}

    def _threshold(self, modality: str) -> float:
        if modality == "text":
            return self.config.classifier.text_threshold
        if modality == "audio":
            return self.config.classifier.audio_threshold
        if modality == "image":
            return self.config.classifier.image_threshold
        return 0.4
