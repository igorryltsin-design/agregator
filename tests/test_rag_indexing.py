import textwrap

from agregator.rag.embeddings import LmStudioEmbeddingBackend, load_embedding_backend
from agregator.rag.indexing import (
    ChunkConfig,
    chunk_text,
    detect_language,
    extract_keywords,
    normalize_text,
)


def test_normalize_text_strips_and_deduplicates():
    raw = "  Первая строка  \r\nВторая строка\n\n\nТретья  "
    normalized = normalize_text(raw)
    assert normalized == "Первая строка\nВторая строка\n\nТретья"


def test_detect_language_handles_ru_en_and_mixed():
    assert detect_language("Привет, мир!") == "ru"
    assert detect_language("Hello world") == "en"
    assert detect_language("Hola こんにちは") in {"mixed", "unknown", "en"}


def test_extract_keywords_drops_stopwords():
    text = "The quick brown fox jumps over the lazy dog and the dog sleeps"
    keywords = extract_keywords(text, lang="en", limit=5)
    assert "the" not in keywords  # stop-word
    assert len(keywords) <= 5


def test_chunk_text_respects_overlap_and_min_tokens():
    sample = " ".join(f"token{i}" for i in range(60))
    cfg = ChunkConfig(max_tokens=20, overlap=5, min_tokens=10)
    chunks = chunk_text(sample, config=cfg)
    assert len(chunks) == 4
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]


def test_load_embedding_backend_lm_studio_instantiates():
    backend = load_embedding_backend(
        "lm-studio",
        model_name="nomic-ai/nomic-embed-text-v1.5-GGUF",
        base_url="http://localhost:1234/v1",
        api_key=None,
    )
    assert isinstance(backend, LmStudioEmbeddingBackend)
    assert backend.model_version.lower().startswith("lm-studio")
