from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable


HEADING_RE = re.compile(
    r"^(?P<kind>Chương|Mục|Điều|Khoản|Điểm)\s+(?P<number>[\w.]+)[:.\s]*(?P<title>.*)$",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"\S+")


def clean_text(value: str | None) -> str:
    value = value or ""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_question(value: str) -> str:
    value = strip_accents(clean_text(value).lower())
    return re.sub(r"[^\w]+", " ", value).strip()


def strip_accents(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    return "".join(
        char
        for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )


def tokenize_words(value: str) -> list[str]:
    return WORD_RE.findall(value)


def tokenize_lexical(value: str, accentless: bool = False) -> list[str]:
    if accentless:
        value = strip_accents(value)
    return re.findall(r"[\w/-]+", value.lower(), flags=re.UNICODE)


def split_sections(text: str) -> Iterable[tuple[dict[str, str], str]]:
    """Yield legal sections while carrying the closest chapter/article headings."""
    headings: dict[str, str] = {}
    buffer: list[str] = []

    def emit() -> tuple[dict[str, str], str] | None:
        content = clean_text("\n".join(buffer))
        return (dict(headings), content) if content else None

    for raw_line in clean_text(text).splitlines():
        line = raw_line.strip()
        match = HEADING_RE.match(line)
        if match and match.group("kind").lower() in {"chương", "mục", "điều"}:
            item = emit()
            if item:
                yield item
            buffer = [line]
            kind = match.group("kind").capitalize()
            headings[kind] = line
            if kind == "Chương":
                headings.pop("Mục", None)
                headings.pop("Điều", None)
            elif kind == "Mục":
                headings.pop("Điều", None)
            continue
        buffer.append(raw_line)
    item = emit()
    if item:
        yield item


def chunk_section(
    section: str,
    metadata_prefix: str,
    max_tokens: int,
    overlap: int,
) -> list[str]:
    if not tokenize_words(section):
        return [metadata_prefix] if metadata_prefix else []
    payload_limit = max(16, max_tokens - len(tokenize_words(metadata_prefix)))
    step = max(1, payload_limit - overlap)
    chunks: list[str] = []

    def emit(words: list[str]) -> None:
        if words:
            chunks.append(clean_text(f"{metadata_prefix}\n{' '.join(words)}"))

    # Prefer natural Khoản/Điểm boundaries. A very long clause still falls back
    # to a sliding window, preserving the configured overlap.
    units = re.split(r"(?=^\s*(?:\d+\.\s|[a-zđ]\)\s))", section, flags=re.IGNORECASE | re.MULTILINE)
    window: list[str] = []
    for unit in units:
        unit_words = tokenize_words(unit)
        if not unit_words:
            continue
        if len(unit_words) > payload_limit:
            if window:
                emit(window)
                window = []
            for start in range(0, len(unit_words), step):
                emit(unit_words[start : start + payload_limit])
                if start + payload_limit >= len(unit_words):
                    break
            continue
        if window and len(window) + len(unit_words) > payload_limit:
            emit(window)
            window = window[-overlap:] if overlap else []
        window.extend(unit_words)
    emit(window)
    return chunks


def legal_chunks(
    document_name: str | None,
    passage: str | None,
    short_tokens: int,
    short_overlap: int,
    long_tokens: int,
    long_overlap: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    short: list[dict[str, str]] = []
    long: list[dict[str, str]] = []
    title = clean_text(document_name) or "Văn bản pháp luật không có tiêu đề"
    for headings, section in split_sections(passage or ""):
        hierarchy = " | ".join([title, *headings.values()])
        short.extend(
            {"text": chunk, "heading": hierarchy}
            for chunk in chunk_section(section, hierarchy, short_tokens, short_overlap)
        )
        long.extend(
            {"text": chunk, "heading": hierarchy}
            for chunk in chunk_section(section, hierarchy, long_tokens, long_overlap)
        )
    if not short:
        fallback = clean_text(f"{title}\n{passage or ''}")
        short = [{"text": fallback, "heading": title}]
        long = [{"text": fallback, "heading": title}]
    return short, long
