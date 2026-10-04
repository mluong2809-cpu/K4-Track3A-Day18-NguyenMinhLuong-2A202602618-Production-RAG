from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY
from src.gemini_backend import chat

_API_AVAILABLE = True


def _ask_gemini(system: str, user: str, json_mode: bool = False) -> str | None:
    """Make one bounded API request; after a failure, use local fallbacks."""
    global _API_AVAILABLE
    if not GEMINI_API_KEY or not _API_AVAILABLE:
        return None
    try:
        return chat(system, user, json_mode=json_mode)
    except Exception as error:
        _API_AVAILABLE = False
        print(f"  Gemini enrichment unavailable: {error}")
        return None


def _fallback_summary(text: str) -> str:
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    first = _fallback_summary(text).split(".")[0].strip()
    return [f"Nội dung nào được quy định về {first[:80]}?"] if first and n_questions > 0 else []


def _fallback_metadata(text: str) -> dict:
    return {"topic": _fallback_summary(text)[:80], "entities": [],
            "category": "policy", "language": "vi"}


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str, *, use_api: bool = False) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    response = _ask_gemini("Tóm tắt đoạn văn trong 2 câu ngắn gọn bằng tiếng Việt.", text) if use_api else None
    return response or _fallback_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3, *, use_api: bool = False) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    response = _ask_gemini(
        f"Tạo {n_questions} câu hỏi mà đoạn văn trả lời được. Mỗi câu hỏi một dòng.",
        text,
    ) if use_api else None
    if response:
        questions = [re.sub(r'^\s*\d+[.)]\s*', '', line).strip()
                     for line in response.splitlines() if line.strip()]
        return questions[:n_questions]
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "", *, use_api: bool = False) -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    response = _ask_gemini(
        "Viết một câu ngắn mô tả vị trí và chủ đề của đoạn văn trong tài liệu.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
    ) if use_api else None
    prefix = response or (f"Trích từ {document_title}." if document_title else "")
    return f"{prefix}\n\n{text}" if prefix else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str, *, use_api: bool = False) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    response = _ask_gemini(
        'Trả về JSON với topic, entities (mảng), category, language.',
        text, json_mode=True,
    ) if use_api else None
    if response:
        try:
            value = json.loads(response)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str, *, use_api: bool = True) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    response = _ask_gemini(
        "Phân tích đoạn văn và trả về một JSON object gồm chính xác bốn trường: "
        "summary (tóm tắt ngắn), questions (mảng 3 câu hỏi), "
        "context (một câu nêu vị trí và chủ đề trong tài liệu), "
        "metadata (object có topic, entities, category, language).",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
        json_mode=True,
    ) if use_api else None
    if response:
        try:
            value = json.loads(response)
            if isinstance(value, dict) and all(key in value for key in
                                               ("summary", "questions", "context", "metadata")):
                return value
        except json.JSONDecodeError:
            pass
    return {
        "summary": _fallback_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": f"Trích từ {source}." if source else "",
        "metadata": _fallback_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
    *,
    use_api: bool = False,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source, use_api=use_api)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text, use_api=use_api) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text, use_api=use_api) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source, use_api=use_api) if "contextual" in methods else text
            auto_meta = extract_metadata(text, use_api=use_api) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
