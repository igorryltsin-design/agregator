import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Создаём лёгкие заглушки для fitz/pymupdf, чтобы не требовать системные библиотеки при импортировании app.py в тестах.
for module_name in ['fitz', 'pymupdf', 'pymupdf.extra', 'pymupdf._extra', 'pymupdf._mupdf']:
    if module_name not in sys.modules:
        sys.modules[module_name] = types.ModuleType(module_name)

import app


def test_ensure_russian_text_translates(monkeypatch):
    calls = {'called': False}

    def fake_compose(system, user, temperature=0.0, max_tokens=0):
        calls['called'] = True
        assert "Переведи" in system
        assert "Исходный текст" in user
        return "Переведённый текст"

    monkeypatch.setattr(app, 'call_lmstudio_compose', fake_compose)
    translated, flag = app._ensure_russian_text("This is a test string.", label='unit-test')
    assert flag is True
    assert translated == "Переведённый текст"
    assert calls['called'] is True


def test_ensure_russian_text_skips_ru(monkeypatch):
    def failing_compose(*args, **kwargs):
        raise AssertionError("Translator should not be called for Russian text")

    monkeypatch.setattr(app, 'call_lmstudio_compose', failing_compose)
    translated, flag = app._ensure_russian_text("Это уже русский текст.", label='unit-test-skip')
    assert translated == "Это уже русский текст."
    assert flag is False
