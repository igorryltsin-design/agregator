import hashlib
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String, unique=True, nullable=False)
    password_hash = db.Column(db.String, nullable=False)
    role = db.Column(db.String, nullable=False, default="user")
    full_name = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password.strip())

    def check_password(self, password: str) -> bool:
        if not password:
            return False
        try:
            return check_password_hash(self.password_hash, password)
        except Exception:
            return False

class Collection(db.Model):
    __tablename__ = "collections"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)
    slug = db.Column(db.String, unique=True, nullable=False)
    searchable = db.Column(db.Boolean, default=True, nullable=False)
    graphable = db.Column(db.Boolean, default=True, nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    is_private = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    owner = db.relationship("User", backref="collections", lazy=True)


class CollectionMember(db.Model):
    __tablename__ = "collection_members"
    id = db.Column(db.Integer, primary_key=True)
    collection_id = db.Column(db.Integer, db.ForeignKey("collections.id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    role = db.Column(db.String, nullable=False, default="viewer")  # допустимые роли: viewer (просмотр) или editor (редактирование)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    collection = db.relationship("Collection", backref="members", lazy=True)
    user = db.relationship("User", backref="collection_memberships", lazy=True)


class ChangeLog(db.Model):
    __tablename__ = "changelog"
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="CASCADE"), index=True, nullable=True)
    action = db.Column(db.String, nullable=False)
    field = db.Column(db.String, nullable=True)
    old_value = db.Column(db.String, nullable=True)
    new_value = db.Column(db.String, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    info = db.Column(db.String, nullable=True)


class File(db.Model):
    __tablename__ = "files"
    id = db.Column(db.Integer, primary_key=True)
    collection_id = db.Column(db.Integer, db.ForeignKey("collections.id", ondelete="SET NULL"), index=True, nullable=True)
    path = db.Column(db.String, unique=True, nullable=False)
    rel_path = db.Column(db.String, nullable=False)
    filename = db.Column(db.String, nullable=False)
    ext = db.Column(db.String, nullable=True)
    size = db.Column(db.Integer, nullable=True)
    mtime = db.Column(db.Float, nullable=True)
    sha1 = db.Column(db.String, nullable=True, index=True)

    material_type = db.Column(db.String, nullable=True)
    title = db.Column(db.String, nullable=True)
    author = db.Column(db.String, nullable=True)
    year = db.Column(db.String, nullable=True)
    advisor = db.Column(db.String, nullable=True)
    keywords = db.Column(db.String, nullable=True)
    abstract = db.Column(db.Text, nullable=True)
    text_excerpt = db.Column(db.Text, nullable=True)

    collection = db.relationship("Collection", backref="files", lazy=True)
    tags = db.relationship("Tag", backref="file", cascade="all, delete-orphan")
    rag_document = db.relationship("RagDocument", backref="file", uselist=False, cascade="all, delete-orphan")
    doc_chat_cache = db.relationship("DocChatCache", backref="file", uselist=False, cascade="all, delete-orphan")


class Tag(db.Model):
    __tablename__ = "tags"
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="CASCADE"), index=True, nullable=False)
    key = db.Column(db.String, index=True, nullable=False)
    value = db.Column(db.String, index=True, nullable=False)


class TagSchema(db.Model):
    __tablename__ = "tag_schemas"
    id = db.Column(db.Integer, primary_key=True)
    material_type = db.Column(db.String, index=True, nullable=False)
    key = db.Column(db.String, index=True, nullable=False)
    description = db.Column(db.String, nullable=True)


class UserActionLog(db.Model):
    __tablename__ = "user_action_log"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    action = db.Column(db.String, nullable=False)
    entity = db.Column(db.String, nullable=True)
    entity_id = db.Column(db.Integer, nullable=True)
    detail = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="action_logs", lazy=True)


class TaskRecord(db.Model):
    __tablename__ = "task_records"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    status = db.Column(db.String, nullable=False, default="pending")
    payload = db.Column(db.Text, nullable=True)
    progress = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    error = db.Column(db.String, nullable=True)


class DocChatCache(db.Model):
    __tablename__ = "doc_chat_cache"
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="CASCADE"), primary_key=True)
    document_id = db.Column(db.Integer, nullable=False)
    data = db.Column(db.JSON, nullable=False, default=dict)
    chunk_count = db.Column(db.Integer, nullable=False, default=0)
    image_count = db.Column(db.Integer, nullable=False, default=0)
    text_size = db.Column(db.Integer, nullable=True)
    embedding_backend = db.Column(db.String, nullable=True)
    embedding_model = db.Column(db.String, nullable=True)
    embedding_dim = db.Column(db.Integer, nullable=True)
    vision_enabled = db.Column(db.Boolean, nullable=False, default=False)
    file_mtime = db.Column(db.Float, nullable=True)
    file_sha1 = db.Column(db.String, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LlmEndpoint(db.Model):
    __tablename__ = "llm_endpoints"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, unique=True, nullable=False)
    base_url = db.Column(db.String, nullable=False)
    model = db.Column(db.String, nullable=False)
    api_key = db.Column(db.String, nullable=True)
    weight = db.Column(db.Float, nullable=False, default=1.0)
    purpose = db.Column(db.String, nullable=True)  # например rerank, summary, transcription
    provider = db.Column(db.String, nullable=False, default='openai', server_default='openai')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AiWordAccess(db.Model):
    __tablename__ = "aiword_access"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    granted_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", foreign_keys=[user_id], backref="aiword_permission", lazy=True)
    granted_by_user = db.relationship("User", foreign_keys=[granted_by], lazy=True)


class AiSearchSnippetCache(db.Model):
    __tablename__ = "ai_search_snippet_cache"
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="CASCADE"), index=True, nullable=False)
    query_hash = db.Column(db.String(64), index=True, nullable=False)
    llm_variant = db.Column(db.Boolean, nullable=False, default=False)
    snippet = db.Column(db.Text, nullable=False)
    meta = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=True, index=True)

    file = db.relationship("File", backref="snippet_cache_entries", lazy=True)
    __table_args__ = (
        db.UniqueConstraint('file_id', 'query_hash', 'llm_variant', name='uq_ai_snippet_key'),
    )


class AiSearchKeywordFeedback(db.Model):
    __tablename__ = "ai_search_keyword_feedback"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="SET NULL"), index=True, nullable=True)
    query_hash = db.Column(db.String(64), index=True, nullable=False)
    keyword = db.Column(db.String(120), nullable=True)
    action = db.Column(db.String(32), nullable=False)  # варианты действий: клик, релевантно, нерелевантно, игнор
    score = db.Column(db.Float, nullable=True)
    detail = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship("User", backref="ai_keyword_feedback", lazy=True)
    file = db.relationship("File", backref="ai_keyword_feedback", lazy=True)


class AiSearchFeedbackModel(db.Model):
    __tablename__ = "ai_search_feedback_model"
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="CASCADE"), primary_key=True)
    positive = db.Column(db.Integer, nullable=False, default=0)
    negative = db.Column(db.Integer, nullable=False, default=0)
    clicks = db.Column(db.Integer, nullable=False, default=0)
    weight = db.Column(db.Float, nullable=False, default=0.0)
    last_positive_at = db.Column(db.DateTime, nullable=True)
    last_negative_at = db.Column(db.DateTime, nullable=True)
    last_click_at = db.Column(db.DateTime, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    file = db.relationship("File", backref="ai_feedback_model", lazy=True)


class AiSearchMetric(db.Model):
    __tablename__ = "ai_search_metrics"
    id = db.Column(db.Integer, primary_key=True)
    query_hash = db.Column(db.String(64), index=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    total_ms = db.Column(db.Integer, nullable=False)
    keywords_ms = db.Column(db.Integer, nullable=True)
    candidate_ms = db.Column(db.Integer, nullable=True)
    deep_ms = db.Column(db.Integer, nullable=True)
    llm_answer_ms = db.Column(db.Integer, nullable=True)
    llm_snippet_ms = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta = db.Column(db.Text, nullable=True)

    user = db.relationship("User", lazy=True)


class RagDocument(db.Model):
    __tablename__ = "rag_documents"
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="CASCADE"), unique=True, nullable=False)
    latest_version = db.Column(db.Integer, nullable=False, default=0)
    is_ready_for_rag = db.Column(db.Boolean, nullable=False, default=False)
    import_status = db.Column(db.String(32), nullable=False, default="pending")
    lang_primary = db.Column(db.String(16), nullable=True)
    last_indexed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    versions = db.relationship(
        "RagDocumentVersion",
        backref="document",
        order_by="RagDocumentVersion.version",
        cascade="all, delete-orphan",
        lazy=True,
    )
    chunks = db.relationship(
        "RagDocumentChunk",
        backref="document",
        cascade="all, delete-orphan",
        lazy=True,
    )


class RagDocumentVersion(db.Model):
    __tablename__ = "rag_document_versions"
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("rag_documents.id", ondelete="CASCADE"), index=True, nullable=False)
    version = db.Column(db.Integer, nullable=False)
    sha256 = db.Column(db.String(64), nullable=True)
    dedupe_hash = db.Column(db.String(64), nullable=True)
    imported_at = db.Column(db.DateTime, default=datetime.utcnow)
    normalizer_version = db.Column(db.String(32), nullable=True)
    metadata_json = db.Column(db.Text, nullable=True)
    raw_text = db.Column(db.Text, nullable=True)
    clean_text = db.Column(db.Text, nullable=True)
    lang_primary = db.Column(db.String(16), nullable=True)
    chunk_count = db.Column(db.Integer, nullable=False, default=0)
    error = db.Column(db.Text, nullable=True)

    chunks = db.relationship(
        "RagDocumentChunk",
        backref="version",
        cascade="all, delete-orphan",
        lazy=True,
    )

    __table_args__ = (
        db.UniqueConstraint("document_id", "version", name="uq_rag_doc_version"),
    )


class RagDocumentChunk(db.Model):
    __tablename__ = "rag_document_chunks"
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey("rag_documents.id", ondelete="CASCADE"), index=True, nullable=False)
    version_id = db.Column(db.Integer, db.ForeignKey("rag_document_versions.id", ondelete="CASCADE"), index=True, nullable=False)
    ordinal = db.Column(db.Integer, nullable=False)
    section_path = db.Column(db.String, nullable=True)
    token_count = db.Column(db.Integer, nullable=True)
    char_count = db.Column(db.Integer, nullable=True)
    content = db.Column(db.Text, nullable=False)
    content_hash = db.Column(db.String(64), nullable=True, index=True)
    preview = db.Column(db.String(512), nullable=True)
    keywords_top = db.Column(db.String(512), nullable=True)
    lang_primary = db.Column(db.String(16), nullable=True)
    meta = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    embeddings = db.relationship(
        "RagChunkEmbedding",
        backref="chunk",
        cascade="all, delete-orphan",
        lazy=True,
    )

    __table_args__ = (
        db.UniqueConstraint("version_id", "ordinal", name="uq_rag_chunk_order"),
    )


class RagChunkEmbedding(db.Model):
    __tablename__ = "rag_chunk_embeddings"
    id = db.Column(db.Integer, primary_key=True)
    chunk_id = db.Column(db.Integer, db.ForeignKey("rag_document_chunks.id", ondelete="CASCADE"), index=True, nullable=False)
    model_name = db.Column(db.String(120), nullable=False)
    model_version = db.Column(db.String(60), nullable=True)
    dim = db.Column(db.Integer, nullable=False)
    vector = db.Column(db.LargeBinary, nullable=False)
    vector_checksum = db.Column(db.String(64), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("chunk_id", "model_name", "model_version", name="uq_rag_chunk_embedding_variant"),
    )


class RagIngestFailure(db.Model):
    __tablename__ = "rag_ingest_failures"
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("files.id", ondelete="SET NULL"), index=True, nullable=True)
    stage = db.Column(db.String(64), nullable=False)
    error = db.Column(db.Text, nullable=False)
    resolved = db.Column(db.Boolean, nullable=False, default=False)
    resolved_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    meta = db.Column(db.Text, nullable=True)

    file = db.relationship("File", lazy=True)


class RagSession(db.Model):
    __tablename__ = "rag_sessions"
    id = db.Column(db.Integer, primary_key=True)
    query = db.Column(db.Text, nullable=False)
    query_lang = db.Column(db.String(16), nullable=True)
    chunk_ids = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=False)
    user_prompt = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)
    validation = db.Column(db.Text, nullable=True)
    model_name = db.Column(db.String(120), nullable=True)
    params = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


def upsert_tag(file_obj: File, key: str, value: str):
    key = (key or "").strip()
    value = (value or "").strip()
    if not key or not value:
        return
    t = Tag.query.filter_by(file_id=file_obj.id, key=key, value=value).first()
    if not t:
        t = Tag(file_id=file_obj.id, key=key, value=value)
        db.session.add(t)


def file_to_dict(f: File):
    return {
        "id": f.id,
        "collection_id": getattr(f, 'collection_id', None),
        "title": f.title,
        "author": f.author,
        "year": f.year,
        "material_type": f.material_type,
        "filename": f.filename,
        "keywords": f.keywords,
        "tags": [{"key": t.key, "value": t.value} for t in f.tags],
        "rel_path": f.rel_path,
        "path": f.path,
        "size": f.size,
        "mtime": f.mtime,
        "sha1": f.sha1,
    }
