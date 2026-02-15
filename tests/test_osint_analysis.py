import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agregator.osint.service import LocalSourceOptions, OsintSearchService
from agregator.osint.storage import OsintRepository, OsintRepositoryConfig


def test_analysis_context_truncates_and_marks_fallback():
    snapshot = {
        "sources": [
            {
                "source": "google",
                "metadata": {"label": "Google", "fallback": True},
                "llm_payload": "",
                "results": [
                    {
                        "title": "Первый результат",
                        "url": "https://example.com/item",
                        "snippet": "a" * 120,
                    },
                    {
                        "title": "Второй результат",
                        "url": "https://example.com/other",
                        "snippet": "b" * 40,
                    },
                ],
            }
        ]
    }
    contexts = OsintSearchService._analysis_context(snapshot, max_items=2, snippet_limit=60)
    assert len(contexts) == 1
    ctx = contexts[0]
    assert ctx["label"] == "Google"
    assert ctx["fallback"] is True
    assert "Режим fallback" in ctx["alerts"][0]
    assert ctx["notes"] == []
    assert len(ctx["entries"]) == 2
    assert ctx["entries"][0]["snippet"].endswith("…")
    assert len(ctx["entries"][0]["snippet"]) <= 60


def test_analysis_messages_include_sources():
    contexts = [
        {
            "label": "Google",
            "fallback": False,
            "entries": [
                {
                    "title": "Result A",
                    "snippet": "Short summary",
                    "url": "https://example.com/a",
                }
            ],
        },
        {
            "label": "Yandex",
            "fallback": True,
            "alerts": ["Режим fallback — требуется ручная проверка."],
            "entries": [],
        },
    ]
    messages = OsintSearchService._analysis_messages("query text", contexts)
    assert messages[0]["role"] == "system"
    assert "аналитик OSINT" in messages[0]["content"]
    user_message = messages[1]["content"]
    assert "Контекст по источникам" in user_message
    assert "Result A" in user_message
    assert "Yandex [проверка вручную]" in user_message
    assert "Режим fallback" in user_message
    assert "Структурированные данные (JSON)" in user_message


def test_analysis_fallback_reports_reason():
    contexts = [
        {
            "label": "Google",
            "fallback": False,
            "entries": [
                {
                    "title": "Result A",
                    "snippet": "Insight",
                    "url": "https://example.com/a",
                }
            ],
        }
    ]
    summary = OsintSearchService._analysis_fallback(
        "оригинальный запрос",
        contexts,
        reason="LLM недоступен",
    )
    assert "оригинальный запрос" in summary
    assert "Result A — Insight (https://example.com/a)" in summary
    assert "LLM недоступен" in summary
    assert summary.endswith("Изучите карточки источников для подробностей.")


def test_combine_results_merges_sources():
    snapshot = {
        "sources": [
            {
                "source": "google",
                "engine": "google",
                "metadata": {"label": "Google"},
                "results": [
                    {"title": "Item A", "url": "https://example.com/a", "rank": 1},
                    {"title": "Item B", "url": "https://example.com/b", "rank": 2},
                ],
            },
            {
                "source": "yandex",
                "engine": "yandex",
                "metadata": {"label": "Yandex"},
                "results": [
                    {"title": "Item A duplicate", "url": "https://example.com/a", "rank": 1},
                    {"title": "Local doc", "metadata": {"path": "/files/123"}, "rank": 3},
                ],
            },
        ]
    }
    combined = OsintSearchService._combine_results(snapshot)
    assert len(combined) == 3
    urls = {item.get("url") for item in combined}
    assert "https://example.com/a" in urls
    merged = next(item for item in combined if item.get("url") == "https://example.com/a")
    assert len(merged["sources"]) == 2
    local_entry = next(item for item in combined if (item.get("metadata") or {}).get("path") == "/files/123")
    assert local_entry["title"] == "Local doc"


def test_local_filesystem_search_reads_content(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text(
        "Строчка без запроса\nА здесь есть секретный план запуска. Связь: team@example.com",
        encoding="utf-8",
    )
    service = OsintSearchService(fetcher=object(), parser=object(), repository=object())
    options = LocalSourceOptions(mode="filesystem", path=str(tmp_path), limit=5, recursive=True)
    payload = service._local_filesystem_search(query="секретный", keywords=["секретный"], options=options, label="Тестовая папка")
    items = payload["items"]
    assert items
    matched = next(item for item in items if item["metadata"]["match"] == "content")
    assert "секретный" in (matched["snippet"] or "").lower()
    assert matched.get("highlight") and "<mark>" in matched["highlight"]
    extracted = matched["metadata"].get("extracted")
    assert extracted and "emails" in extracted and "team@example.com" in extracted["emails"]


def test_local_filesystem_search_reads_documents(monkeypatch, tmp_path):
    doc_path = tmp_path / "report.pdf"
    doc_path.write_bytes(b"%PDF-1.4 fake")

    def fake_extract(path: Path, suffix: str, *, limit: int, force_ocr: bool = False) -> str:
        assert suffix == ".pdf"
        return "Это секретный отчёт о проекте. Email: archive@example.org"

    monkeypatch.setattr(
        OsintSearchService,
        "_filesystem_extract_document",
        staticmethod(fake_extract),
    )
    service = OsintSearchService(fetcher=object(), parser=object(), repository=object())
    options = LocalSourceOptions(mode="filesystem", path=str(tmp_path), limit=5, recursive=True)
    payload = service._local_filesystem_search(query="секретный", keywords=["секретный"], options=options, label="Документы")
    items = payload["items"]
    assert items
    matched = next(item for item in items if item["metadata"]["match"] == "content")
    assert "секретный" in (matched["snippet"] or "").lower()
    assert matched["metadata"].get("extracted")


def test_local_filesystem_exclude_patterns(tmp_path):
    (tmp_path / "secret-plan.txt").write_text("секретный план", encoding="utf-8")
    (tmp_path / "public.txt").write_text("открытые данные", encoding="utf-8")
    service = OsintSearchService(fetcher=object(), parser=object(), repository=object())
    options = LocalSourceOptions(
        mode="filesystem",
        path=str(tmp_path),
        limit=5,
        recursive=True,
        exclude_patterns=("*secret*",),
    )
    payload = service._local_filesystem_search(query="секретный", keywords=["секретный"], options=options, label="Документы")
    assert all("secret" not in item["metadata"]["path"] for item in payload["items"])


def test_highlight_text_marks_terms():
    highlight = OsintSearchService._highlight_text("Пример секретного плана", ["секретного"])
    assert highlight is not None
    assert "<mark>секретного</mark>" in highlight.lower()

def test_refine_query_filters_stopwords():
    service = OsintSearchService(fetcher=object(), parser=object(), repository=object())
    refined, keywords = service._refine_query('что такое NL2SQL и как это работает')
    assert 'nl2sql' in keywords
    assert 'что' not in keywords
    assert refined


def test_export_job_markdown_contains_sources():
    repo = OsintRepository(OsintRepositoryConfig(url="sqlite:///:memory:"))
    service = OsintSearchService(fetcher=object(), parser=object(), repository=repo)
    job = repo.create_job(
        query="финансирование исследований",
        locale="ru-RU",
        region=None,
        safe=False,
        sources=[{"type": "engine", "engine": "google", "id": "google"}],
        params={"keywords": ["финансирование", "исследования"]},
        user_id=None,
    )
    repo.persist_source_result(
        job_id=job["id"],
        source_id="google",
        engine="google",
        blocked=False,
        from_cache=False,
        html_snapshot="<html>demo</html>",
        text_content="demo",
        screenshot_path=None,
        llm_payload=None,
        llm_model=None,
        llm_error=None,
        status="completed",
        error=None,
        metadata={"label": "Google"},
        results=[
            {
                "rank": 1,
                "title": "Demo Result",
                "url": "https://example.com",
                "snippet": "Описание",
                "metadata": {},
            }
        ],
    )
    repo.set_job_analysis(job["id"], "Наблюдения:\n- пример", None)
    markdown = service.export_job_markdown(job["id"])
    assert "OSINT-отчёт" in markdown
    assert "Demo Result" in markdown
    assert "Google" in markdown


def test_persist_source_result_stores_artifacts(tmp_path):
    repo = OsintRepository(OsintRepositoryConfig(url="sqlite:///:memory:"))
    job = repo.create_job(
        query="проверка",
        locale="ru-RU",
        region=None,
        safe=False,
        sources=[{"type": "engine", "engine": "google", "id": "google"}],
        params={},
        user_id=None,
    )
    repo.persist_source_result(
        job_id=job["id"],
        source_id="google",
        engine="google",
        blocked=False,
        from_cache=False,
        html_snapshot="<html><body>hello</body></html>",
        text_content="hello",
        screenshot_path="screenshots/demo.png",
        llm_payload=None,
        llm_model=None,
        llm_error=None,
        status="completed",
        error=None,
        metadata={"label": "Google"},
        results=[
            {
                "rank": 1,
                "title": "Demo",
                "url": "https://example.com",
                "snippet": "Свяжитесь с нами: team@example.com",
                "metadata": {
                    "highlight": "<mark>Свяжитесь</mark>",
                    "extracted": {"emails": ["team@example.com"]},
                },
            }
        ],
    )
    stored = repo.get_job(job["id"])
    assert stored["sources"][0]["text_content"] == "hello"
    assert stored["sources"][0]["screenshot_path"] == "screenshots/demo.png"
    result_meta = stored["sources"][0]["results"][0]["metadata"]
    assert result_meta.get("highlight")
    assert result_meta.get("extracted") and "team@example.com" in result_meta["extracted"].get("emails", [])


def test_extract_structured_data():
    text = "Контакты: media@example.com и +1 (555) 123-4567"
    structured = OsintSearchService._extract_structured(text)
    assert structured["emails"] == ["media@example.com"]
    assert any("555" in phone for phone in structured["phones"])


def test_analysis_llm_notes_from_payload():
    source = {
        "llm_payload": json.dumps(
            {
                "items": [
                    {"title": "Report", "snippet": "Key finding", "url": "https://example.com/report"},
                    {"title": "Blog", "snippet": "Another note"},
                ]
            }
        )
    }
    notes = OsintSearchService._analysis_llm_notes(source, max_items=2)
    assert len(notes) == 2
    assert "Report — Key finding" in notes[0]
