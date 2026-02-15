"""Filesystem walker for multimodal scanner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from sqlalchemy.orm import Session

from ..config import AppConfig
from ..repositories import ArtifactRepository, LogRepository, RunRepository, SignatureRepository, SourceRepository
from .. import utils

LOGGER = logging.getLogger("scanner.walker")


class ScanWalker:
    def __init__(self, config: AppConfig):
        self.config = config

    def run(self, session: Session, memo: str | None = None) -> dict:
        run_repo = RunRepository(session)
        source_repo = SourceRepository(session)
        artifact_repo = ArtifactRepository(session)
        signature_repo = SignatureRepository(session)
        log_repo = LogRepository(session)

        run = run_repo.start_run(memo)
        total = processed = errors = 0
        for root in self.config.source.root_paths:
            root_path = Path(root).expanduser()
            if not root_path.exists():
                log_repo.add(run.id, "warning", "Root path missing", {"path": str(root_path)})
                continue
            source = source_repo.get_or_create(
                path=str(root_path),
                root=str(root_path),
                recursive=True,
                patterns=self.config.source.include_patterns,
            )
            for file_path in utils.iter_files(
                root_path,
                max_depth=self.config.source.max_depth,
                follow_symlinks=self.config.source.follow_symlinks,
                patterns=self.config.source.include_patterns,
            ):
                total += 1
                try:
                    modality = utils.guess_modality(file_path)
                    signature = utils.rolling_hash(
                        file_path, self.config.fingerprint.block_size, self.config.fingerprint.hash_alg
                    )
                    entropy = utils.entropy(file_path.read_bytes()[:4096])
                    attributes = {
                        "extension": file_path.suffix.lower(),
                        "parent": str(file_path.parent),
                    }
                    artifact = artifact_repo.upsert(
                        source_id=source.id,
                        path=str(file_path),
                        size_bytes=file_path.stat().st_size,
                        modality=modality,
                        signature=signature,
                        hash_alg=self.config.fingerprint.hash_alg,
                        entropy=entropy,
                        attributes=attributes,
                    )
                    blocks = utils.chunk_signatures(
                        file_path,
                        block_size=self.config.fingerprint.block_size,
                        hash_alg=self.config.fingerprint.hash_alg,
                    )
                    signature_repo.replace(artifact.id, blocks)
                    processed += 1
                except Exception as exc:
                    errors += 1
                    log_repo.add(
                        run.id,
                        "error",
                        "Failed to process file",
                        {"path": str(file_path), "error": str(exc)},
                    )
        run_repo.finish_run(run.id, total=total, processed=processed, errors=errors)
        LOGGER.info("Scan finished: total=%s processed=%s errors=%s", total, processed, errors)
        return {"run_id": run.id, "total": total, "processed": processed, "errors": errors}
