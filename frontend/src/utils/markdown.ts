import DOMPurify from 'dompurify'

const ALLOWED_TAGS = [
  'a',
  'b',
  'strong',
  'em',
  'i',
  'code',
  'pre',
  'p',
  'br',
  'ul',
  'ol',
  'li',
  'span',
  'div',
  'blockquote',
  'hr',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
]
const ALLOWED_ATTR = ['href', 'title', 'target', 'rel', 'class']

let dompurifyConfigured = false
const configureDomPurify = () => {
  if (dompurifyConfigured) return
  if (typeof window === 'undefined') return
  DOMPurify.addHook('afterSanitizeAttributes', node => {
    if (node.tagName === 'A') {
      const href = node.getAttribute('href') || ''
      if (/^javascript:/i.test(href)) {
        node.removeAttribute('href')
      }
      if (node.getAttribute('target') === '_blank') {
        node.setAttribute('rel', 'noopener noreferrer')
      }
    }
    node.removeAttribute('style')
  })
  dompurifyConfigured = true
}

const escapeHtml = (value: string): string =>
  value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

const applyInlineFormatting = (value: string): string => {
  let result = escapeHtml(value)
  result = result.replace(/`([^`]+)`/g, (_, code) => `<code>${code.trim()}</code>`)
  result = result.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  result = result.replace(/__([^_]+)__/g, '<strong>$1</strong>')
  result = result.replace(/\*(?!\s)([^*]+)\*/g, '<em>$1</em>')
  result = result.replace(/_(?!\s)([^_]+)_/g, '<em>$1</em>')
  result = result.replace(/~~([^~]+)~~/g, '<del>$1</del>')
  result = result.replace(
    /\[([^\]]+)]\(([^)]+)\)/g,
    (_, text, href) => `<a href="${href}" target="_blank" rel="noopener">${text}</a>`,
  )
  return result
}

const closeLists = (state: { inUl: boolean; inOl: boolean }, buffer: string[]): void => {
  if (state.inUl) {
    buffer.push('</ul>')
    state.inUl = false
  }
  if (state.inOl) {
    buffer.push('</ol>')
    state.inOl = false
  }
}

const handleListItem = (
  line: string,
  state: { inUl: boolean; inOl: boolean },
  buffer: string[],
  ordered: boolean,
): void => {
  const content = line.replace(ordered ? /^\d+\.\s*/ : /^[-*+]\s*/, '')
  if (ordered) {
    if (!state.inOl) {
      closeLists(state, buffer)
      buffer.push('<ol>')
      state.inOl = true
    }
  } else if (!state.inUl) {
    closeLists(state, buffer)
    buffer.push('<ul>')
    state.inUl = true
  }
  buffer.push(`<li>${applyInlineFormatting(content)}</li>`)
}

const handleParagraph = (
  line: string,
  state: { inUl: boolean; inOl: boolean },
  buffer: string[],
): void => {
  closeLists(state, buffer)
  buffer.push(`<p>${applyInlineFormatting(line)}</p>`)
}

const handleHeading = (
  line: string,
  level: number,
  state: { inUl: boolean; inOl: boolean },
  buffer: string[],
): void => {
  closeLists(state, buffer)
  const content = line.replace(/^#+\s*/, '')
  const safeLevel = Math.min(Math.max(level, 1), 6)
  buffer.push(`<h${safeLevel}>${applyInlineFormatting(content)}</h${safeLevel}>`)
}

const handleBlockquote = (
  line: string,
  state: { inUl: boolean; inOl: boolean },
  buffer: string[],
): void => {
  closeLists(state, buffer)
  const content = line.replace(/^>\s?/, '')
  buffer.push(`<blockquote>${applyInlineFormatting(content)}</blockquote>`)
}

const handleHorizontalRule = (
  state: { inUl: boolean; inOl: boolean },
  buffer: string[],
): void => {
  closeLists(state, buffer)
  buffer.push('<hr />')
}

const markdownToHtml = (markdown: string): string => {
  const lines = markdown.split(/\r?\n/)
  const buffer: string[] = []
  const state = { inUl: false, inOl: false, inCode: false }
  let codeBuffer: string[] = []

  for (const rawLine of lines) {
    const line = rawLine
    const trimmed = line.trim()

    if (state.inCode) {
      if (trimmed.startsWith('```')) {
        buffer.push(`<pre><code>${escapeHtml(codeBuffer.join('\n'))}</code></pre>`)
        codeBuffer = []
        state.inCode = false
      } else {
        codeBuffer.push(line)
      }
      continue
    }

    if (trimmed.startsWith('```')) {
      closeLists(state, buffer)
      state.inCode = true
      codeBuffer = []
      continue
    }

    if (!trimmed) {
      closeLists(state, buffer)
      continue
    }

    if (/^#{1,6}\s/.test(trimmed)) {
      const level = trimmed.match(/^#+/)?.[0].length || 1
      handleHeading(trimmed, level, state, buffer)
      continue
    }

    if (/^[-*+]\s+/.test(trimmed)) {
      handleListItem(trimmed, state, buffer, false)
      continue
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      handleListItem(trimmed, state, buffer, true)
      continue
    }

    if (/^>\s?/.test(trimmed)) {
      handleBlockquote(trimmed, state, buffer)
      continue
    }

    if (/^(-{3,}|_{3,}|\*{3,})$/.test(trimmed)) {
      handleHorizontalRule(state, buffer)
      continue
    }

    handleParagraph(trimmed, state, buffer)
  }

  closeLists(state, buffer)
  if (state.inCode && codeBuffer.length) {
    buffer.push(`<pre><code>${escapeHtml(codeBuffer.join('\n'))}</code></pre>`)
  }

  return buffer.join('\n')
}

export const renderMarkdown = (markdown: string | null | undefined): string => {
  if (!markdown) return ''
  const html = markdownToHtml(markdown)
  if (typeof window === 'undefined') {
    return html
  }
  configureDomPurify()
  return DOMPurify.sanitize(html, { ALLOWED_TAGS, ALLOWED_ATTR })
}
