from flask import Flask

from agregator.rag import (
    ChunkConfig,
    ContextSelector,
    ContextSection,
    KeywordRetriever,
    RagIndexer,
    VectorRetriever,
    build_system_prompt,
    build_user_prompt,
    detect_language,
    load_embedding_backend,
    validate_answer,
)
from models import db, File, RagDocumentChunk, RagChunkEmbedding
from sqlalchemy import and_
from hashlib import sha256
from array import array


def _create_app():
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db.init_app(app)
    return app


def _seed_file():
    text = (
        "Обеспечение безопасности данных в блокчейн-системе "
        "требует применения современных методов машинного обучения."
    )
    file_obj = File(
        path="dummy.pdf",
        rel_path="dummy.pdf",
        filename="dummy.pdf",
        title="Безопасность данных",
        text_excerpt=text,
    )
    db.session.add(file_obj)
    db.session.commit()
    return file_obj


def test_rag_pipeline_end_to_end():
    app = _create_app()
    with app.app_context():
        db.create_all()
        file_obj = _seed_file()

        indexer = RagIndexer(chunk_config=ChunkConfig(max_tokens=32, overlap=8, min_tokens=8))
        result = indexer.ingest_document(
            file_obj,
            file_obj.text_excerpt or "",
            metadata={"source": "unit-test"},
            commit=True,
        )
        assert result["chunks"] > 0

        backend = load_embedding_backend("hash", model_name="test-hash", dim=64)

        join_condition = and_(
            RagChunkEmbedding.chunk_id == RagDocumentChunk.id,
            RagChunkEmbedding.model_name == backend.model_name,
            RagChunkEmbedding.model_version == backend.model_version,
        )
        batch = (
            db.session.query(RagDocumentChunk)
            .outerjoin(RagChunkEmbedding, join_condition)
            .filter(RagChunkEmbedding.id.is_(None))
            .all()
        )
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

        vector_retriever = VectorRetriever(
            model_name=backend.model_name,
            model_version=backend.model_version,
            max_candidates=20,
        )
        keyword_retriever = KeywordRetriever(limit=20)
        selector = ContextSelector(
            vector_retriever=vector_retriever,
            keyword_retriever=keyword_retriever,
            dense_weight=1.0,
            sparse_weight=0.5,
            max_per_document=2,
        )

        query = "обеспечение безопасности данных"
        query_vec = backend.embed_many([query])[0]
        contexts = selector.select(query, query_vec, top_k=2)
        assert contexts, "Context selector should return at least one chunk"

        sections = [
        ContextSection(
            doc_id=cand.chunk.document_id,
            chunk_id=cand.chunk.id,
            title=file_obj.title or "Документ",
            language=cand.chunk.lang_primary or detect_language(cand.chunk.content or ""),
            score_dense=cand.dense_score,
            score_sparse=cand.sparse_score,
            combined_score=cand.combined_score,
            reasoning_hint=cand.reasoning_hint,
            preview=getattr(cand, "preview", cand.chunk.preview or ""),
            content=(cand.chunk.content or "")[:400],
        )
            for cand in contexts
        ]
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(query, sections)
        assert "Факты" in system_prompt
        assert "Контекст" in user_prompt

        fallback = validate_answer("Факты:\n- Источников не найдено\nИсточники:\n- Источников не найдено", [])
        assert fallback.is_empty is False
        assert fallback.missing_citations is True
