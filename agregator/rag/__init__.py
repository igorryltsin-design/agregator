"""RAG indexing utilities."""

from .context import ContextCandidate, ContextSelector
from .embeddings import (
    EmbeddingBackend,
    HashEmbeddingBackend,
    SentenceTransformersBackend,
    load_embedding_backend,
)
from .indexing import (
    ChunkConfig,
    RagIndexer,
    chunk_text,
    detect_language,
    extract_keywords,
    normalize_text,
)
from .prompt import ContextSection, build_system_prompt, build_user_prompt, fallback_answer
from .retrieval import RetrievedChunk, VectorRetriever
from .sparse import KeywordMatch, KeywordRetriever
from .utils import bytes_to_vector, vector_to_bytes
from .validation import ValidationResult, extract_citations, validate_answer
from .rerank import CrossEncoderConfig, CrossEncoderReranker, load_reranker

__all__ = [
    "ChunkConfig",
    "RagIndexer",
    "chunk_text",
    "detect_language",
    "extract_keywords",
    "normalize_text",
    "EmbeddingBackend",
    "HashEmbeddingBackend",
    "SentenceTransformersBackend",
    "load_embedding_backend",
    "vector_to_bytes",
    "bytes_to_vector",
    "VectorRetriever",
    "RetrievedChunk",
    "KeywordRetriever",
    "KeywordMatch",
    "ContextSelector",
    "ContextCandidate",
    "ContextSection",
    "build_system_prompt",
    "build_user_prompt",
    "fallback_answer",
    "ValidationResult",
    "extract_citations",
    "validate_answer",
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "load_reranker",
]
