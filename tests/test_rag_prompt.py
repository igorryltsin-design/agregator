import re

from agregator.rag import (
    ContextSection,
    build_system_prompt,
    build_user_prompt,
    fallback_answer,
    validate_answer,
)


def test_build_system_prompt_default():
    system = build_system_prompt()
    assert "Факты" in system
    assert "Источники" in system
    assert "Источников не найдено" in system


def test_build_user_prompt_renders_sections():
    sections = [
        ContextSection(
            doc_id=10,
            chunk_id=42,
            title="Документ А",
            language="en",
            translation_hint="Фрагмент на en, ответ оставь на русском, цитаты — на оригинале.",
            score_dense=0.91,
            score_sparse=0.37,
            combined_score=1.28,
            reasoning_hint="dense=0.91; keywords=ai; sparse=0.37",
            preview="Краткое описание",
            content="Полный текст чанка",
            url="library/doc_a.pdf",
            extra={"section_path": "§1.2"},
        )
    ]
    prompt = build_user_prompt("Что такое RAG?", sections)
    assert "Вопрос пользователя" in prompt
    assert "[1] doc_id=10 chunk_id=42" in prompt
    assert "lang=en" in prompt
    assert "Краткое описание" in prompt
    assert "Полный текст чанка" in prompt


def test_fallback_answer_returns_template():
    fb = fallback_answer()
    assert "Источников не найдено" in fb
    assert fb.count("Источников") == 2


def test_validate_answer_success():
    answer = (
        "Факты:\n"
        "- Тестовый факт [10:40]\n"
        "Источники:\n"
        "- [10:40] Документ А (ru)\n"
    )
    result = validate_answer(answer, allowed_references=[(10, 40)])
    assert result.hallucination_warning is False
    assert not result.missing_citations
    assert not result.unknown_citations


def test_validate_answer_detects_issues():
    answer = (
        "Факты:\n"
        "- Факт без ссылки\n"
        "- Факт с неизвестной ссылкой [7:1]\n"
        "Источники:\n"
        "- [99:99] Заглушка\n"
    )
    result = validate_answer(answer, allowed_references=[(5, 5)])
    assert result.hallucination_warning is True
    assert result.missing_citations is True
    assert (7, 1) in result.unknown_citations
    assert result.extra_citations == [(99, 99)]
    assert any(re.search("Факт без", line) for line in result.facts_with_issues)
