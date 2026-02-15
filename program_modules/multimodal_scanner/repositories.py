"""Repository layer for scanner module."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import models


class BaseRepository:
    def __init__(self, session: Session):
        self.session = session


class SourceRepository(BaseRepository):
    def get_or_create(self, path: str, root: str, recursive: bool, patterns: List[str]) -> models.ScanSource:
        stmt = select(models.ScanSource).where(models.ScanSource.path == path)
        source = self.session.scalar(stmt)
        if source:
            return source
        source = models.ScanSource(
            path=path,
            root=root,
            recursive=recursive,
            include_patterns=patterns,
        )
        self.session.add(source)
        self.session.flush()
        return source


class ArtifactRepository(BaseRepository):
    def upsert(
        self,
        source_id: int,
        path: str,
        size_bytes: int,
        modality: str,
        signature: str,
        hash_alg: str,
        entropy: float,
        attributes: Dict,
    ) -> models.Artifact:
        stmt = select(models.Artifact).where(models.Artifact.signature == signature)
        artifact = self.session.scalar(stmt)
        if artifact:
            artifact.path = path
            artifact.size_bytes = size_bytes
            artifact.modality = modality
            artifact.hash_alg = hash_alg
            artifact.entropy = entropy
            artifact.attributes = attributes
            artifact.updated_at = datetime.utcnow()
        else:
            artifact = models.Artifact(
                source_id=source_id,
                path=path,
                size_bytes=size_bytes,
                modality=modality,
                signature=signature,
                hash_alg=hash_alg,
                entropy=entropy,
                attributes=attributes,
            )
            self.session.add(artifact)
            self.session.flush()
        return artifact

    def list_recent(self, limit: int = 50) -> List[models.Artifact]:
        stmt = (
            select(models.Artifact)
            .order_by(models.Artifact.updated_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class SignatureRepository(BaseRepository):
    def replace(self, artifact_id: int, blocks: Sequence[tuple[int, str, int]]) -> None:
        self.session.query(models.SignatureBlock).filter(
            models.SignatureBlock.artifact_id == artifact_id
        ).delete(synchronize_session=False)
        for offset, signature, block_size in blocks:
            block = models.SignatureBlock(
                artifact_id=artifact_id,
                offset=offset,
                signature=signature,
                block_size=block_size,
            )
            self.session.add(block)

    def load(self, artifact_id: int) -> List[models.SignatureBlock]:
        stmt = select(models.SignatureBlock).where(models.SignatureBlock.artifact_id == artifact_id)
        return list(self.session.scalars(stmt))


class SimilarityRepository(BaseRepository):
    def upsert(self, a_id: int, b_id: int, score: float, reason: str) -> models.SimilarityEdge:
        if a_id == b_id:
            raise ValueError("Cannot link artifact to itself")
        key = tuple(sorted((a_id, b_id)))
        stmt = select(models.SimilarityEdge).where(
            models.SimilarityEdge.from_artifact_id == key[0],
            models.SimilarityEdge.to_artifact_id == key[1],
        )
        edge = self.session.scalar(stmt)
        if edge:
            edge.score = score
            edge.reason = reason
            edge.created_at = datetime.utcnow()
        else:
            edge = models.SimilarityEdge(
                from_artifact_id=key[0],
                to_artifact_id=key[1],
                score=score,
                reason=reason,
            )
            self.session.add(edge)
            self.session.flush()
        return edge

    def neighbors(self, artifact_id: int, limit: int = 20) -> List[models.SimilarityEdge]:
        stmt = (
            select(models.SimilarityEdge)
            .where(
                (models.SimilarityEdge.from_artifact_id == artifact_id)
                | (models.SimilarityEdge.to_artifact_id == artifact_id)
            )
            .order_by(models.SimilarityEdge.score.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class RunRepository(BaseRepository):
    def start_run(self, memo: str | None = None) -> models.ScanRun:
        run = models.ScanRun(started_at=datetime.utcnow(), memo=memo)
        self.session.add(run)
        self.session.flush()
        return run

    def finish_run(self, run_id: int, total: int, processed: int, errors: int) -> None:
        run = self.session.get(models.ScanRun, run_id)
        if run:
            run.finished_at = datetime.utcnow()
            run.total_files = total
            run.processed_files = processed
            run.errors = errors


class LogRepository(BaseRepository):
    def add(self, run_id: int, level: str, message: str, payload: Dict | None = None) -> None:
        entry = models.ScanLog(
            run_id=run_id,
            level=level,
            message=message,
            payload=payload or {},
        )
        self.session.add(entry)

    def latest(self, limit: int = 50) -> List[models.ScanLog]:
        stmt = (
            select(models.ScanLog)
            .order_by(models.ScanLog.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


class StatRepository(BaseRepository):
    def store(self, metric: str, window_start: datetime, window_end: datetime, payload: Dict) -> None:
        stat = models.ScanStat(
            metric_name=metric,
            window_start=window_start,
            window_end=window_end,
            payload=payload,
        )
        self.session.add(stat)

    def recent(self, metric: str, limit: int = 10) -> List[models.ScanStat]:
        stmt = (
            select(models.ScanStat)
            .where(models.ScanStat.metric_name == metric)
            .order_by(models.ScanStat.window_end.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))


def modality_stats(session: Session, window: timedelta) -> Dict[str, int]:
    since = datetime.utcnow() - window
    stmt = (
        select(models.Artifact.modality, func.count(models.Artifact.id))
        .where(models.Artifact.updated_at >= since)
        .group_by(models.Artifact.modality)
    )
    return dict(session.execute(stmt).all())
