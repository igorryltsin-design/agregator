import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agregator.osint.markdown import render_markdown


def test_render_markdown_basic_blocks():
    md = """## Заголовок

**Жирный** и *курсив* и `код`

- Пункт 1
- Пункт 2

1. Первый
2. Второй

> Цитата

---

[Ссылка](https://example.com) и [опасная](javascript:alert(1))

```python
print("hello")
```
"""
    html = render_markdown(md)
    assert "<h2>Заголовок</h2>" in html
    assert "<strong>Жирный</strong>" in html
    assert "<em>курсив</em>" in html
    assert "<code>код</code>" in html
    assert "<ul>" in html and html.count("<li>") >= 4
    assert "<ol>" in html
    assert "<blockquote>" in html
    assert "<hr />" in html
    assert '<a href="https://example.com"' in html
    assert "javascript" not in html
    assert "print(&quot;hello&quot;)" in html


def test_render_markdown_empty():
    assert render_markdown("") == ""
    assert render_markdown(None) == ""
