"""Simple index builder that extracts keywords from artifact metadata."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models


class KeywordIndex:
    TOKEN_RE = re.compile(r"[a-zA-Z0-9а-яА-Я]{3,}")

    def __init__(self):
        self.index: Dict[str, List[int]] = defaultdict(list)

    def build(self, session: Session, limit: int = 500) -> Dict[str, List[int]]:
        self.index.clear()
        rows = session.scalars(
            select(models.Artifact).order_by(models.Artifact.updated_at.desc()).limit(limit)
        ).all()
        for artifact in rows:
            tokens = self._tokens(artifact)
            for token in tokens:
                posting = self.index[token]
                if artifact.id not in posting:
                    posting.append(artifact.id)
        return self.index

    def search(self, query: str) -> List[int]:
        token = query.lower()
        return self.index.get(token, [])

    def _tokens(self, artifact: models.Artifact) -> List[str]:
        attrs = artifact.attributes or {}
        text = f"{artifact.path} {attrs.get('extension', '')}"
        return [match.group(0).lower() for match in self.TOKEN_RE.finditer(text)]
