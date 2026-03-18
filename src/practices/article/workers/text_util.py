"""PDF 文本处理工具（精简版）。"""

from __future__ import annotations

import re
from typing import List, Tuple


def detect_table_pattern_regions(lines: List[str]) -> List[Tuple[int, int]]:
    """检测可能的表格区块（短行 + 连续出现）。"""

    def is_table_like(line: str) -> bool:
        line = line.strip()
        if not line or len(line) > 50:
            return False
        if re.match(r"^(TABLE|FIGURE|QUADRANT)\s+[IVX0-9]+", line, re.IGNORECASE):
            return True
        if re.match(r"^[\d\s.,]+$", line):
            return True
        if line.isupper() and len(line) < 30:
            return True
        if len(line.split()) <= 3 and len(line) < 30:
            return True
        if "\t" in line or re.search(r"\s{3,}", line):
            return True
        return False

    regions: List[Tuple[int, int]] = []
    start = None
    length = 0

    for idx, line in enumerate(lines):
        if is_table_like(line):
            if start is None:
                start = idx
                length = 1
            else:
                length += 1
            continue

        if start is not None and length >= 3:
            regions.append((start, idx - 1))
        start = None
        length = 0

    if start is not None and length >= 3:
        regions.append((start, len(lines) - 1))

    return regions


def extract_text_with_positions(pdf_data: bytes, page_num: int) -> Tuple[str, List[float]]:
    """提取页文本和每行 y 坐标。"""
    try:
        import fitz  # PyMuPDF，按需导入，不影响不依赖 PDF 的流程
        with fitz.open(stream=pdf_data, filetype="pdf") as doc:
            if page_num >= len(doc):
                return "", []

            page = doc[page_num]
            blocks = page.get_text("dict").get("blocks", [])
            lines: List[str] = []
            positions: List[float] = []

            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    if line_text.strip():
                        lines.append(line_text)
                        positions.append(line["bbox"][1])

            return "\n".join(lines), positions
    except Exception:
        return "", []


def smart_restore_paragraphs(text: str) -> str:
    """智能还原段落，保留页码/元信息标记。"""
    lines = text.split("\n")
    total_lines = len(lines)
    table_regions = detect_table_pattern_regions(lines)

    special_markers = {
        "abstract",
        "keywords",
        "introduction",
        "conclusion",
        "references",
        "acknowledgments",
        "appendix",
        "results",
        "discussion",
        "methods",
        "methodology",
    }
    sentence_endings = (".", "!", "?", "。", "！", "？")
    continue_punctuation = (",", ";", ":", "，", "；", "：")
    page_number_found = False

    def in_table(line_idx: int) -> bool:
        for start, end in table_regions:
            if start <= line_idx <= end:
                return True
        return False

    def is_table_marker(line: str) -> bool:
        line_upper = line.upper().strip()
        return bool(
            re.match(r"^(TABLE|FIGURE)\s+\d+", line_upper)
            or re.match(r"^QUADRANT\s+[IVX]+", line_upper)
        )

    def in_page_zone(line_idx: int) -> bool:
        return line_idx < 5 or line_idx >= total_lines - 5

    def is_page_number(line: str, line_idx: int, line_length: int) -> bool:
        nonlocal page_number_found
        if page_number_found or line_length > 50 or not in_page_zone(line_idx):
            return False

        line = line.strip()
        if re.match(r"^\d{1,4}$", line) or re.match(
            r"^(\d+[-/]\d+|Page\s+\d+|\d+\s*of\s*\d+)$", line, re.IGNORECASE
        ):
            page_number_found = True
            return True
        return False

    def is_special_line(line: str) -> bool:
        line_lower = line.lower().strip()
        if is_table_marker(line):
            return True
        if line.isupper() and 10 < len(line) < 100:
            return True
        return any(line_lower.startswith(marker) for marker in special_markers)

    def is_header_footer(line: str, line_idx: int, line_length: int, is_table_line: bool) -> bool:
        if line_length > 50 or is_table_line or not in_page_zone(line_idx):
            return False

        line = line.strip()
        if len(line) < 5:
            return True

        patterns = [
            r"^\d+$",
            r"^Page\s+\d+",
            r"^\d+\s+\|",
            r"wileyonlinelibrary\.com",
            r"©.*\d{4}",
            r"^[A-Z\s]+\d{4};",
        ]
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in patterns)

    def should_merge(prev_line: str, current_line: str, is_table_line: bool) -> bool:
        if not prev_line or not current_line:
            return False

        prev = prev_line.strip()
        curr = current_line.strip()

        if is_table_line:
            return prev.endswith("-")
        if prev.endswith("-"):
            return True
        if prev[-1] in continue_punctuation:
            return True
        if curr and curr[0].islower():
            return True
        if prev[-1] not in sentence_endings:
            if not (is_special_line(curr) or is_page_number(curr, 0, len(curr)) or is_header_footer(curr, 0, len(curr), False)):
                return True
        return False

    paragraphs: List[str] = []
    current_paragraph: List[str] = []
    prev_line = ""

    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        line_len = len(line)
        is_table_line = in_table(idx) and line_len <= 50

        if not line:
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            prev_line = ""
            continue

        if is_page_number(line, idx, line_len):
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            paragraphs.append(f"[Journal Page: {line}]")
            prev_line = ""
            continue

        if is_header_footer(line, idx, line_len, is_table_line):
            if "©" in line or "DOI" in line.upper():
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(f"[Metadata: {line}]")
                prev_line = ""
            continue

        if is_special_line(line):
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
                current_paragraph = []
            paragraphs.append(line)
            prev_line = line
            continue

        if should_merge(prev_line, line, is_table_line) and current_paragraph:
            if prev_line.endswith("-"):
                current_paragraph[-1] = current_paragraph[-1][:-1] + line
            else:
                current_paragraph.append(line)
        else:
            if current_paragraph and not is_table_line:
                if prev_line and prev_line[-1] in sentence_endings and line and not line[0].islower():
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
            current_paragraph.append(line)

        prev_line = line

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return "\n\n".join(paragraphs)


def restore_pdf_page_text(pdf_data: bytes, page_num: int) -> str:
    """高层接口：提取单页文本并做智能段落还原。"""
    text, _ = extract_text_with_positions(pdf_data, page_num)

    if not text:
        try:
            import fitz  # PyMuPDF，按需导入
            with fitz.open(stream=pdf_data, filetype="pdf") as doc:
                if page_num < len(doc):
                    text = doc[page_num].get_text()
        except Exception:
            return ""

    return smart_restore_paragraphs(text)
