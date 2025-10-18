#!/usr/bin/env python3
"""
Утилита управления RAG-индексом.

Примеры:
    python scripts/rag_index.py rebuild --max-chunks 40
    python scripts/rag_index.py ingest --file-id 15 --file-id 27
    python scripts/rag_index.py inspect --chunk 10
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from array import array
from pathlib import Path

from flask import Flask

from agregator.rag import ChunkConfig, RagIndexer, load_embedding_backend
from models import (
    File,
    RagChunkEmbedding,
    RagDocument,
    RagDocumentChunk,
    db,
)
from sqlalchemy import and_

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "catalogue.db"


def create_app():
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{DB_PATH}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    return app


def ingest_documents(app, file_ids: list[int] | None, *, max_tokens: int, overlap: int, min_tokens: int) -> int:
    cfg = ChunkConfig(max_tokens=max_tokens, overlap=overlap, min_tokens=min_tokens)
    indexer = RagIndexer(chunk_config=cfg)
    processed = 0
    with app.app_context():
        query = File.query.filter(File.text_excerpt.isnot(None))
        if file_ids:
            query = query.filter(File.id.in_(file_ids))
        files = query.all()
        for file_obj in files:
            res = indexer.ingest_document(
                file_obj,
                file_obj.text_excerpt or "",
                metadata={"source": "rag-index cli"},
                skip_if_unchanged=True,
                commit=True,
            )
            if not res.get("skipped"):
                processed += 1
    return processed


def embed_chunks(
    app,
    *,
    backend_name: str,
    model_name: str,
    dim: int,
    batch_size: int,
    normalize: bool = True,
    device: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> int:
    backend = load_embedding_backend(
        backend_name,
        model_name=model_name,
        dim=dim,
        normalize=normalize,
        batch_size=batch_size,
        device=device,
        base_url=endpoint,
        api_key=api_key,
        timeout=timeout,
    )
    total = 0
    try:
        with app.app_context():
            while True:
                join_condition = and_(
                    RagChunkEmbedding.chunk_id == RagDocumentChunk.id,
                    RagChunkEmbedding.model_name == backend.model_name,
                    RagChunkEmbedding.model_version == backend.model_version,
                )
                batch = (
                    db.session.query(RagDocumentChunk)
                    .outerjoin(RagChunkEmbedding, join_condition)
                    .filter(RagChunkEmbedding.id.is_(None))
                    .limit(batch_size)
                    .all()
                )
                if not batch:
                    break
                vectors = backend.embed_many([chunk.content or "" for chunk in batch])
                for chunk, vector in zip(batch, vectors):
                    arr = array("f", vector)
                    vector_bytes = arr.tobytes()
                    checksum = sha256(vector_bytes).hexdigest()
                    db.session.add(
                        RagChunkEmbedding(
                            chunk_id=chunk.id,
                            model_name=backend.model_name,
                            model_version=backend.model_version,
                            dim=len(vector),
                            vector=vector_bytes,
                            vector_checksum=checksum,
                        )
                    )
                db.session.commit()
                total += len(batch)
    finally:
        try:
            backend.close()
        except Exception:
            pass
    return total


def command_rebuild(args):
    app = create_app()
    ingested = ingest_documents(app, args.file_id, max_tokens=args.max_tokens, overlap=args.overlap, min_tokens=args.min_tokens)
    embedded = embed_chunks(
        app,
        backend_name=args.backend,
        model_name=args.model_name,
        dim=args.dim,
        batch_size=args.batch_size,
        normalize=args.normalize,
        device=args.device,
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    print(f"Ingested documents: {ingested}, embeddings updated: {embedded}")


def command_ingest(args):
    app = create_app()
    count = ingest_documents(app, args.file_id, max_tokens=args.max_tokens, overlap=args.overlap, min_tokens=args.min_tokens)
    print(f"Ingested documents: {count}")


def command_embed(args):
    app = create_app()
    count = embed_chunks(
        app,
        backend_name=args.backend,
        model_name=args.model_name,
        dim=args.dim,
        batch_size=args.batch_size,
        normalize=args.normalize,
        device=args.device,
        endpoint=args.endpoint,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    print(f"Embeddings generated for chunks: {count}")


def command_inspect(args):
    app = create_app()
    with app.app_context():
        chunk = RagDocumentChunk.query.get(args.chunk)
        if not chunk:
            print(f"Chunk {args.chunk} not found")
            return
        doc = RagDocument.query.get(chunk.document_id)
        file_obj = getattr(doc, "file", None)
        print(f"Chunk {chunk.id} (doc {doc.id})")
        if file_obj:
            print(f"  File: {file_obj.title or file_obj.filename}")
        print(f"  Token count: {chunk.token_count}")
        print(f"  Language: {chunk.lang_primary}")
        print(f"  Preview: {chunk.preview[:200] if chunk.preview else ''}")
        embedding = RagChunkEmbedding.query.filter_by(chunk_id=chunk.id).first()
        if embedding:
            print(f"  Embedding: model={embedding.model_name} dim={embedding.dim}")
        else:
            print("  Embedding: отсутствует")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Управление RAG-индексом.")
    sub = parser.add_subparsers(dest="command", required=True)

    common_ingest = argparse.ArgumentParser(add_help=False)
    common_ingest.add_argument("--file-id", type=int, action="append", help="Индексация только указанных file_id (можно повторять).")
    common_ingest.add_argument("--max-tokens", type=int, default=700)
    common_ingest.add_argument("--overlap", type=int, default=120)
    common_ingest.add_argument("--min-tokens", type=int, default=80)

    common_embed = argparse.ArgumentParser(add_help=False)
    common_embed.add_argument("--backend", default="hash", choices=["hash", "auto", "sentence-transformers", "lm-studio", "openai"])
    common_embed.add_argument("--model-name", default="intfloat/multilingual-e5-large")
    common_embed.add_argument("--dim", type=int, default=384)
    common_embed.add_argument("--batch-size", type=int, default=256)
    common_embed.add_argument("--device", default=None)
    common_embed.add_argument("--endpoint", default=None, help="URL OpenAI/LM Studio embeddings API")
    common_embed.add_argument("--api-key", default=None, help="API ключ для embeddings API")
    common_embed.add_argument("--timeout", type=float, default=None)
    common_embed.add_argument("--no-normalize", dest="normalize", action="store_false")
    common_embed.set_defaults(normalize=True)

    rebuild = sub.add_parser("rebuild", parents=[common_ingest, common_embed], help="Переиндексировать документы и пересчитать эмбеддинги.")
    rebuild.set_defaults(func=command_rebuild)

    ingest = sub.add_parser("ingest", parents=[common_ingest], help="Проиндексировать документы (без пересчёта эмбеддингов).")
    ingest.set_defaults(func=command_ingest)

    embed = sub.add_parser("embed", parents=[common_embed], help="Пересчитать эмбеддинги для всех чанков.")
    embed.set_defaults(func=command_embed)

    inspect = sub.add_parser("inspect", help="Посмотреть содержимое чанка.")
    inspect.add_argument("--chunk", type=int, required=True)
    inspect.set_defaults(func=command_inspect)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not DB_PATH.exists():
        raise SystemExit(f"catalogue.db not found at {DB_PATH}")
    args.func(args)


if __name__ == "__main__":
    main()
