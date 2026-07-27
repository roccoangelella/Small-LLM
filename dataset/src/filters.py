"""Conservative deterministic prose, English, and code-dump filters."""

from __future__ import annotations

import re

from dataset import config

from .models import TextMetrics


CODE_LINE_RE = re.compile(
    r"""^\s*(?:
        (?:from\s+[\w.]+\s+)?import\s+[\w.*{}, ]+|
        (?:async\s+)?def\s+\w+\s*\(|class\s+\w+[(:]|function\s+\w+\s*\(|
        (?:const|let|var|public|private|protected|static|final)\s+[\w<>,\[\]]+\s+\w+\s*[=;(]|
        (?:\#include|using\s+namespace|namespace|package|interface|enum|struct)\b|
        (?:func|fn)\s+\w+\s*\(|(?:if|for|while|switch)\s*\([^\n]*\)\s*\{|
        \$\s*(?:npm|pip|python|git|cargo|make)\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)
FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
API_MARKER_RE = re.compile(
    r"\b(?:api\s+reference|parameters?|arguments?|returns?|raises?|examples?|methods?|attributes?|see\s+also|endpoint|request\s+body|response\s+body)\b",
    re.IGNORECASE,
)
REPOSITORY_PATH_RE = re.compile(
    r"(?:^|[\s`'\"])(?:[\w.-]+/){1,8}[\w.-]+\.(?:"
    + "|".join(config.CODE_FILE_EXTENSIONS)
    + r")(?:$|[\s`'\",:)])",
    re.IGNORECASE | re.MULTILINE,
)
SOURCE_FILE_TITLE_RE = re.compile(
    r"^\s*(?:file:\s*)?[\w./-]+\.(?:" + "|".join(config.CODE_FILE_EXTENSIONS) + r")\s*$",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z]+")
ENGLISH_MARKERS = frozenset(
    "the be to of and a in that have i it for not on with he as you do at this but his by "
    "from they we say her she or an will my one all would there their what so up out if about "
    "who get which go me when make can like time no just him know take people into year your good "
    "some could them see other than then now look only come its over think also back after use two how "
    "our work first well way even new want because any these give day most us is are was were been being "
    "may should text document information research study data system".split()
)


def _is_code_line(line: str) -> bool:
    """Return whether a line has a source-code-like shape."""

    stripped = line.strip()
    if not stripped:
        return False
    if CODE_LINE_RE.search(stripped):
        return True
    punctuation = sum(character in "{}[]();=<>" for character in stripped)
    compact = max(1, len(re.sub(r"\s+", "", stripped)))
    return punctuation / compact >= 0.18 and bool(re.search(r"[{};=]", stripped))


def inspect_text(text: str) -> TextMetrics:
    """Measure code/API-dump and English signals without an LLM classifier."""

    character_count = len(text)
    lines = [line for line in text.splitlines() if line.strip()]
    line_count = len(lines)
    words = WORD_RE.findall(text.lower())
    word_count = len(words)
    code_line_count = sum(_is_code_line(line) for line in lines)
    code_line_fraction = code_line_count / max(1, line_count)
    fenced_code_characters = sum(len(match.group(0)) for match in FENCED_CODE_RE.finditer(text))
    fenced_code_fraction = fenced_code_characters / max(1, character_count)
    code_symbols = sum(character in "{}[]();<>`" for character in text)
    code_symbol_fraction = code_symbols / max(1, character_count)
    api_marker_count = len(API_MARKER_RE.findall(text))
    repository_path_count = len(REPOSITORY_PATH_RE.findall(text))
    letters = sum(character.isascii() and character.isalpha() for character in text)
    ascii_letter_ratio = letters / max(1, character_count)
    english_marker_hits = sum(word in ENGLISH_MARKERS for word in words)
    english_marker_ratio = english_marker_hits / max(1, word_count)
    likely_english = (
        ascii_letter_ratio >= config.MIN_ASCII_LETTER_RATIO
        and english_marker_hits >= config.MIN_ENGLISH_MARKER_HITS
        and english_marker_ratio >= config.MIN_ENGLISH_MARKER_RATIO
    )

    first_nonempty = lines[0] if lines else ""
    source_file_title = bool(SOURCE_FILE_TITLE_RE.fullmatch(first_nonempty))
    code_dominated = False
    rejection_reason: str | None = None
    if source_file_title and line_count >= config.MIN_LINES_FOR_CODE_FRACTION:
        code_dominated, rejection_reason = True, "source_file_title"
    elif fenced_code_fraction >= config.MAX_FENCED_CODE_FRACTION:
        code_dominated, rejection_reason = True, "fenced_code_fraction"
    elif (
        line_count >= config.MIN_LINES_FOR_CODE_FRACTION
        and code_line_count >= config.MIN_CODE_LINES_FOR_REJECTION
        and code_line_fraction >= config.MAX_CODE_LINE_FRACTION
    ):
        code_dominated, rejection_reason = True, "code_line_fraction"
    elif (
        code_symbol_fraction >= config.MAX_CODE_SYMBOL_FRACTION
        and code_line_fraction >= config.MAX_CODE_LINE_FRACTION * 0.65
    ):
        code_dominated, rejection_reason = True, "code_symbol_fraction"
    elif (
        api_marker_count >= config.MIN_API_MARKERS_FOR_REJECTION
        and code_line_fraction >= config.MAX_CODE_LINE_FRACTION * 0.4
    ):
        code_dominated, rejection_reason = True, "generated_api_reference"
    elif (
        repository_path_count >= config.MIN_REPOSITORY_PATHS_FOR_REJECTION
        and code_line_fraction >= config.MAX_CODE_LINE_FRACTION * 0.4
    ):
        code_dominated, rejection_reason = True, "repository_or_code_dump"

    return TextMetrics(
        character_count=character_count,
        line_count=line_count,
        word_count=word_count,
        code_line_count=code_line_count,
        code_line_fraction=round(code_line_fraction, 6),
        fenced_code_fraction=round(fenced_code_fraction, 6),
        code_symbol_fraction=round(code_symbol_fraction, 6),
        api_marker_count=api_marker_count,
        repository_path_count=repository_path_count,
        ascii_letter_ratio=round(ascii_letter_ratio, 6),
        english_marker_hits=english_marker_hits,
        english_marker_ratio=round(english_marker_ratio, 6),
        likely_english=likely_english,
        code_dominated=code_dominated,
        rejection_reason=rejection_reason,
    )


def selection_rejection(metrics: TextMetrics) -> str | None:
    """Return the deterministic reason a decoded document is not selectable."""

    if metrics.character_count < config.MIN_DOCUMENT_CHARACTERS:
        return "too_short"
    if metrics.code_dominated:
        return metrics.rejection_reason or "code_dominated"
    if config.REQUIRE_LIKELY_ENGLISH and not metrics.likely_english:
        return "not_likely_english"
    return None


def filter_settings_snapshot() -> dict[str, object]:
    """Store every configurable eligibility setting with a generated artifact."""

    names = (
        "TOKENIZER_ENCODING", "MIN_DOCUMENT_CHARACTERS", "REQUIRE_LIKELY_ENGLISH",
        "MIN_ASCII_LETTER_RATIO", "MIN_ENGLISH_MARKER_RATIO", "MIN_ENGLISH_MARKER_HITS",
        "MAX_CODE_LINE_FRACTION", "MAX_FENCED_CODE_FRACTION", "MAX_CODE_SYMBOL_FRACTION",
        "MIN_CODE_LINES_FOR_REJECTION", "MIN_LINES_FOR_CODE_FRACTION",
        "MIN_API_MARKERS_FOR_REJECTION", "MIN_REPOSITORY_PATHS_FOR_REJECTION",
        "CODE_FILE_EXTENSIONS",
    )
    snapshot: dict[str, object] = {}
    for name in names:
        value = getattr(config, name)
        # JSON turns tuples into lists. Normalizing here keeps a persisted
        # inventory/plan comparable to the live configuration on the next run.
        snapshot[name] = list(value) if isinstance(value, tuple) else value
    return snapshot
