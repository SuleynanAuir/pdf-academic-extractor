#!/usr/bin/env python3
"""Extract OCR text and literature-form metadata with Mistral Document AI."""

from __future__ import annotations

import argparse
import base64
from difflib import SequenceMatcher
import hashlib
import json
import os
import re
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any

try:
    import httpx
    from pydantic import BaseModel, ConfigDict, Field
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit(
        "Missing dependency. Run: python3 -m pip install -r requirements.txt"
    ) from exc


DEFAULT_PDF = Path(
    "ASC_handbook/《耐火材料的损毁及其抑制技术》作者：王诚训 等 .pdf"
)
DEFAULT_OUTPUT_DIR = Path("result/asc_handbook_mistral_experiment_v2")
DEFAULT_ENV_FILE = Path(".env")
OCR_API_URL = "https://api.mistral.ai/v1/ocr"
PAGES_PER_OCR_CALL = 8
METADATA_PAGE_COUNT = 3
MAX_API_ATTEMPTS = 3

ANNOTATION_PROMPT = """
Extract metadata for a Chinese literature-entry form using ONLY text visibly printed
in this document. The selected pages are the cover, title page, and copyright/CIP
page. Follow these rules strictly:

1. Return every schema key. Use null or [] when a value is not explicitly supported.
2. title: the primary work's full title, joining line-wrapped title text. Exclude
   edition statements, running headers, advertisements, series names, and publisher.
3. authors: creators of the primary work only, in printed order. Remove role labels
   such as 著, 编著, 主编, edited by. Preserve original-language names.
4. venue_or_publisher: journal/conference/proceedings/publisher name. For a book,
   return the publisher printed on the title or CIP page.
5. document_type: journal_article, conference_paper, patent, book, or unknown.
6. year: publication year of this edition, not a cited work or earlier edition.
7. pages: article page range/article number; for a book, its explicitly printed total
   page count, without inventing a range.
8. doi: only an explicit DOI beginning with 10.; an ISBN is never a DOI.
9. abstract: use an explicitly printed 摘要/内容提要/Abstract. Do not invent details.
10. note: at most two short sentences. Prefer edition and ISBN information. Do not
    summarize chapters and do not repeat the abstract.
""".strip()


class DocumentType(str, Enum):
    JOURNAL_ARTICLE = "journal_article"
    CONFERENCE_PAPER = "conference_paper"
    PATENT = "patent"
    BOOK = "book"
    UNKNOWN = "unknown"


class LiteratureMetadata(BaseModel):
    """All keys are required; unknown values must be explicit null/empty values."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(
        description=(
            "Main document title. Join line-wrapped title text; exclude edition, "
            "journal running headers, series names, advertisements, and publisher."
        ),
        max_length=500,
    )
    authors: list[str] = Field(
        description=(
            "Primary document authors in printed order, without role words such as "
            "著, 编著, 主编, or edited by."
        ),
        max_length=100,
    )
    venue_or_publisher: str | None = Field(
        description="Journal, conference, proceedings, or book publisher name.",
        max_length=500,
    )
    document_type: DocumentType = Field(
        description="Type of the primary document, based only on explicit evidence."
    )
    year: int | None = Field(
        description="Publication year of this edition; null when not printed.",
        ge=1000,
        le=2100,
    )
    pages: str | None = Field(
        description=(
            "Article page range/article number; for a book, the explicitly printed "
            "total page count, such as 219."
        ),
        max_length=100,
    )
    doi: str | None = Field(
        description="Explicit DOI beginning with 10.; never put an ISBN here.",
        max_length=300,
    )
    abstract: str | None = Field(
        description=(
            "Printed 摘要, 内容提要, or Abstract text; null when the document has none."
        ),
        max_length=4000,
    )
    note: str | None = Field(
        description=(
            "At most two short sentences for edition, ISBN, source, or another "
            "essential qualification. Do not repeat the abstract."
        ),
        max_length=600,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract full OCR Markdown and form-ready bibliographic metadata from "
            "one PDF with Mistral Document AI."
        )
    )
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="mistral-ocr-latest")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to the .env file containing MISTRAL_API_KEY (default: .env).",
    )
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Rebuild derived outputs from existing raw_response_pages_*.json files.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> bool:
    """Load MISTRAL_API_KEY from a small dotenv file without extra dependencies.

    An already exported environment variable takes precedence over the file.
    The parser accepts either ``MISTRAL_API_KEY=value`` or
    ``export MISTRAL_API_KEY=value`` and optional single/double quotes.
    """
    if os.environ.get("MISTRAL_API_KEY"):
        return True
    path = path.expanduser()
    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or key.strip() != "MISTRAL_API_KEY":
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            os.environ["MISTRAL_API_KEY"] = value
            return True
    return False


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def response_format_payload() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": LiteratureMetadata.__name__,
            "schema": LiteratureMetadata.model_json_schema(),
            "strict": True,
        },
    }


def chunk_page_indexes(page_count: int) -> list[list[int]]:
    metadata_end = min(METADATA_PAGE_COUNT, page_count)
    chunks = [list(range(metadata_end))]
    for start in range(metadata_end, page_count, PAGES_PER_OCR_CALL):
        chunks.append(list(range(start, min(start + PAGES_PER_OCR_CALL, page_count))))
    return [chunk for chunk in chunks if chunk]


def process_ocr(
    client: httpx.Client,
    *,
    model: str,
    pdf_data_url: str,
    page_indexes: list[int],
    extract_metadata: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "document": {
            "type": "document_url",
            "document_url": pdf_data_url,
        },
        "pages": page_indexes,
        "table_format": "markdown",
        "include_image_base64": False,
        "include_blocks": False,
        "confidence_scores_granularity": "page",
    }
    if extract_metadata:
        payload["document_annotation_format"] = response_format_payload()
        payload["document_annotation_prompt"] = ANNOTATION_PROMPT

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            response = client.post(OCR_API_URL, json=payload)
        except httpx.HTTPError:
            if attempt == MAX_API_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))
            continue
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise TypeError(f"Expected OCR object response, got {type(result)}")
            return result
        if attempt == MAX_API_ATTEMPTS:
            response.raise_for_status()
        retry_after = response.headers.get("retry-after")
        delay = float(retry_after) if retry_after else float(2 ** (attempt - 1))
        time.sleep(min(delay, 30.0))
    raise RuntimeError("OCR retry loop ended unexpectedly")


def parse_annotation(value: Any) -> LiteratureMetadata:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected document_annotation to be an object or JSON string, got {type(value)}"
        )
    return LiteratureMetadata.model_validate(value)


def markdown_with_inline_tables(page: dict[str, Any]) -> str:
    markdown = str(page.get("markdown") or "")
    for table in page.get("tables") or []:
        table_id = table.get("id")
        content = str(table.get("content") or "").strip()
        if table_id and content:
            markdown = markdown.replace(f"[{table_id}]({table_id})", content)
    return markdown


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_author(value: str) -> str:
    value = normalize_spaces(value)
    value = re.sub(r"\s*(?:编著|主编|著|编|译)$", "", value).strip()
    return re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", value)


def normalize_doi(value: str | None) -> str | None:
    if value is None:
        return None
    value = normalize_spaces(value)
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", value, flags=re.I)
    value = value.rstrip(".,;。；")
    if not re.fullmatch(r"10\.\d{4,9}/\S+", value, flags=re.I):
        return None
    return value


def extract_printed_abstract(
    page_markdown: dict[int, str], metadata_page_count: int
) -> tuple[str | None, list[int]]:
    heading = re.compile(
        r"(?im)^#{1,6}\s*(?:内容提要|内容简介|摘\s*要|abstract)\s*$"
    )
    next_heading = re.compile(r"(?m)^#{1,6}\s+.+$")
    for page_index in range(min(metadata_page_count, len(page_markdown))):
        text = page_markdown.get(page_index, "")
        match = heading.search(text)
        if not match:
            continue
        remainder = text[match.end() :].lstrip()
        stop = next_heading.search(remainder)
        section = (remainder[: stop.start()] if stop else remainder).strip()
        if len(section) >= 20:
            return section, [page_index + 1]
    return None, []


def normalize_metadata(
    metadata: LiteratureMetadata,
    page_markdown: dict[int, str],
    metadata_page_count: int = METADATA_PAGE_COUNT,
) -> tuple[dict[str, Any], list[int]]:
    value = metadata.model_dump(mode="json")
    for key in ("title", "venue_or_publisher", "pages", "note"):
        if value[key] is not None:
            value[key] = normalize_spaces(value[key])

    authors: list[str] = []
    for author in value["authors"]:
        normalized = normalize_author(author)
        if normalized and normalized not in authors:
            authors.append(normalized)
    value["authors"] = authors
    value["doi"] = normalize_doi(value["doi"])

    printed_abstract, abstract_pages = extract_printed_abstract(
        page_markdown, metadata_page_count
    )
    if printed_abstract:
        value["abstract"] = printed_abstract
    else:
        value["abstract"] = None
    return value, abstract_pages


def compact_for_search(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value, flags=re.UNICODE).lower()


def pages_containing(value: str | int | None, page_markdown: dict[int, str]) -> list[int]:
    if value is None:
        return []
    needle = compact_for_search(str(value))
    if not needle:
        return []
    return [
        index + 1
        for index, text in page_markdown.items()
        if needle in compact_for_search(text)
    ]


def isbn13_checksum_valid(value: str) -> bool:
    normalized = re.sub(r"[^\dXx]", "", value)
    if len(normalized) == 13 and normalized.isdigit():
        total = sum(
            int(digit) * (1 if index % 2 == 0 else 3)
            for index, digit in enumerate(normalized[:12])
        )
        return (10 - total % 10) % 10 == int(normalized[-1])
    if len(normalized) == 10 and normalized[:9].isdigit():
        check_value = 10 if normalized[-1].upper() == "X" else int(normalized[-1])
        total = sum((10 - index) * int(digit) for index, digit in enumerate(normalized[:9]))
        total += check_value
        return total % 11 == 0
    return False


def build_field_evidence(
    metadata: dict[str, Any],
    page_markdown: dict[int, str],
    abstract_pages: list[int],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for key in ("title", "venue_or_publisher", "year", "pages", "doi"):
        field_value = metadata[key]
        if (
            key == "pages"
            and metadata["document_type"] == DocumentType.BOOK.value
            and field_value is not None
            and re.fullmatch(r"\d+", str(field_value))
        ):
            page_pattern = re.compile(
                rf"(?<!\d){re.escape(str(field_value))}\s*(?:页|pages?\b|p\.)",
                flags=re.I,
            )
            pages = [
                index + 1
                for index, text in page_markdown.items()
                if page_pattern.search(text)
            ]
        else:
            pages = pages_containing(field_value, page_markdown)
        components: dict[str, list[int]] = {}
        if key == "venue_or_publisher" and field_value and not pages:
            parts = [
                part.strip()
                for part in re.split(
                    r"\s+(?:and|&)\s+|[;；]|\s+(?:与|和)\s+",
                    str(field_value),
                    flags=re.I,
                )
                if len(part.strip()) >= 3
            ]
            components = {
                part: pages_containing(part, page_markdown) for part in parts
            }
            if components and all(components.values()):
                pages = sorted({page for found in components.values() for page in found})
        if field_value is None or field_value == "":
            status = "empty_not_claimed"
        else:
            status = "verified_in_ocr" if pages else "not_found_in_ocr"
        evidence[key] = {
            "status": status,
            "pages": pages,
        }
        if components:
            evidence[key]["pages_by_component"] = components

    author_evidence = {
        author: pages_containing(author, page_markdown) for author in metadata["authors"]
    }
    evidence["authors"] = {
        "status": (
            "verified_in_ocr"
            if author_evidence and all(author_evidence.values())
            else ("partial_or_not_found" if author_evidence else "empty_not_claimed")
        ),
        "pages_by_author": author_evidence,
    }

    type_pages: list[int] = []
    if metadata["document_type"] == DocumentType.BOOK.value:
        for index, text in page_markdown.items():
            if (
                "图书在版编目" in text
                or "ISBN" in text
                or re.search(r"catalog(?:ing|uing)-in-publication", text, flags=re.I)
                or re.search(r"edited by", text, flags=re.I)
            ):
                type_pages.append(index + 1)
    evidence["document_type"] = {
        "status": "verified_in_ocr" if type_pages else "not_independently_verified",
        "pages": sorted(set(type_pages)),
    }
    if metadata.get("abstract") is None:
        abstract_status = "empty_not_claimed"
    else:
        abstract_status = (
            "transcribed_from_printed_section" if abstract_pages else "model_extracted"
        )
    evidence["abstract"] = {"status": abstract_status, "pages": abstract_pages}
    note = metadata.get("note") or ""
    isbn_match = re.search(
        r"(?:ISBN\s*)?((?:97[89][\d-]{10,}|[\dXx][\dXx-]{8,}))",
        note,
        flags=re.I,
    )
    isbn = isbn_match.group(1).rstrip("-;；,.。") if isbn_match else None
    isbn_pages = pages_containing(isbn, page_markdown)
    edition_match = re.search(r"第\s*\d+\s*版", note)
    edition = edition_match.group(0) if edition_match else None
    edition_pages = pages_containing(edition, page_markdown)
    editor_match = re.search(r"edited by\s+([^;,.]+(?:\.[^;,.]+)*)", note, flags=re.I)
    editor = editor_match.group(1).strip() if editor_match else None
    editor_pages = pages_containing(editor, page_markdown)
    note_verified = bool(note) and bool(isbn_pages or edition_pages or editor_pages)
    evidence["note"] = {
        "status": "verified_in_ocr" if note_verified else "review_recommended",
        "pages": sorted(set(isbn_pages + edition_pages + editor_pages)),
        "isbn": isbn,
        "isbn_checksum_valid": isbn13_checksum_valid(isbn) if isbn else None,
    }
    return evidence


def drop_unverified_optional_claims(
    metadata: dict[str, Any], field_evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    """Clear optional identifier/publication claims absent from the selected OCR pages."""
    adjustments: list[dict[str, Any]] = []
    for field in ("venue_or_publisher", "year", "pages", "doi"):
        value = metadata[field]
        if value is None or value == "":
            continue
        if field_evidence[field]["status"] == "not_found_in_ocr":
            adjustments.append(
                {
                    "field": field,
                    "removed_value": value,
                    "reason": "Value was not found in the selected OCR evidence pages.",
                }
            )
            metadata[field] = None
    return adjustments


def correct_cover_title(
    page_markdown: dict[int, str], title: str | None, evidence_pages: list[int]
) -> list[dict[str, Any]]:
    """Correct a near-identical cover heading when later pages verify the title."""
    if not title or 0 not in page_markdown:
        return []
    if len([page for page in evidence_pages if page != 1]) < 2:
        return []

    lines = page_markdown[0].splitlines()
    heading_indexes: list[int] = []
    for index, line in enumerate(lines[:8]):
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"第\s*\d+\s*版|编著|主编|出版社", stripped):
            break
        if stripped.startswith("#"):
            heading_indexes.append(index)
        elif heading_indexes:
            break
    if not heading_indexes:
        return []

    original_parts = [re.sub(r"^#+\s*", "", lines[i]).strip() for i in heading_indexes]
    original_title = "".join(original_parts)
    original_compact = compact_for_search(original_title)
    expected_compact = compact_for_search(title)
    if not original_compact or original_compact == expected_compact:
        return []
    similarity = SequenceMatcher(None, original_compact, expected_compact).ratio()
    if similarity < 0.92:
        return []

    lines[heading_indexes[0]] = f"# {title}"
    for index in reversed(heading_indexes[1:]):
        del lines[index]
    page_markdown[0] = "\n".join(lines)
    return [
        {
            "page": 1,
            "field": "title",
            "original": original_title,
            "corrected": title,
            "similarity": round(similarity, 4),
            "reason": (
                "Cover OCR heading differed by a near-match, while the canonical "
                "title was independently verified on at least two later pages."
            ),
        }
    ]


def build_qa_report(
    page_confidence: list[dict[str, Any]],
    field_evidence: dict[str, Any],
    corrections: list[dict[str, Any]],
) -> dict[str, Any]:
    averages: list[tuple[int, float]] = []
    for item in page_confidence:
        score = (item.get("confidence_scores") or {}).get(
            "average_page_confidence_score"
        )
        if isinstance(score, (int, float)):
            averages.append((int(item["page"]), float(score)))
    low_confidence = [
        {"page": page, "average_page_confidence_score": score}
        for page, score in averages
        if score < 0.95
    ]
    verified_fields = [
        field
        for field, item in field_evidence.items()
        if str(item.get("status", "")).startswith("verified")
        or item.get("status") == "transcribed_from_printed_section"
    ]
    return {
        "average_page_confidence_score": (
            sum(score for _, score in averages) / len(averages) if averages else None
        ),
        "low_confidence_threshold": 0.95,
        "low_confidence_pages": low_confidence,
        "verified_form_fields": verified_fields,
        "corrections": corrections,
        "raw_ocr_preserved": True,
    }


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not args.reuse_raw and not api_key:
        print(
            f"MISTRAL_API_KEY is not set. Add it to {args.env_file} or export it "
            "in the shell, then rerun this command.",
            file=sys.stderr,
        )
        return 2

    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    page_count = len(PdfReader(pdf_path).pages)
    if page_count == 0:
        print(f"PDF has no pages: {pdf_path}", file=sys.stderr)
        return 2

    pdf_bytes = pdf_path.read_bytes()
    pdf_data_url = "data:application/pdf;base64," + base64.b64encode(
        pdf_bytes
    ).decode("ascii")
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_page_indexes(page_count)
    page_markdown: dict[int, str] = {}
    usage_records: list[dict[str, Any]] = []
    page_confidence: list[dict[str, Any]] = []
    raw_annotation: Any = None
    annotation_raw_file: str | None = None

    client: httpx.Client | None = None
    if not args.reuse_raw:
        client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=args.timeout
        )
    try:
        for chunk_number, page_indexes in enumerate(chunks):
            first_page = page_indexes[0] + 1
            last_page = page_indexes[-1] + 1
            extract_metadata = chunk_number == 0
            suffix = " with strict form metadata" if extract_metadata else ""
            raw_path = output_dir / (
                f"raw_response_pages_{first_page:03d}-{last_page:03d}.json"
            )
            if args.reuse_raw:
                print(f"Reusing source pages {first_page}-{last_page}{suffix}...")
                if not raw_path.is_file():
                    raise FileNotFoundError(f"Raw OCR response not found: {raw_path}")
                response = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                print(f"Processing source pages {first_page}-{last_page}{suffix}...")
                if client is None:
                    raise RuntimeError("HTTP client was not initialized")
                response = process_ocr(
                    client,
                    model=args.model,
                    pdf_data_url=pdf_data_url,
                    page_indexes=page_indexes,
                    extract_metadata=extract_metadata,
                )
                write_json(raw_path, response)

            response_pages = response.get("pages") or []
            if len(response_pages) != len(page_indexes):
                raise RuntimeError(
                    f"Expected {len(page_indexes)} OCR pages for source pages "
                    f"{first_page}-{last_page}, got {len(response_pages)}"
                )
            for source_page_index, page in zip(page_indexes, response_pages):
                page_markdown[source_page_index] = markdown_with_inline_tables(page).strip()
                if page.get("confidence_scores") is not None:
                    page_confidence.append(
                        {
                            "page": source_page_index + 1,
                            "confidence_scores": page["confidence_scores"],
                        }
                    )

            if extract_metadata:
                raw_annotation = response.get("document_annotation")
                annotation_raw_file = raw_path.name
            usage = response.get("usage_info")
            if isinstance(usage, dict):
                usage_records.append(
                    {"source_pages": [first_page, last_page], **usage}
                )
    finally:
        if client is not None:
            client.close()

    if raw_annotation is None:
        raise RuntimeError("Mistral response did not include document_annotation")
    annotation = parse_annotation(raw_annotation)
    form_metadata, abstract_pages = normalize_metadata(annotation, page_markdown)
    field_evidence = build_field_evidence(
        form_metadata, page_markdown, abstract_pages
    )
    corrections = correct_cover_title(
        page_markdown,
        form_metadata["title"],
        field_evidence["title"]["pages"],
    )
    if corrections:
        field_evidence = build_field_evidence(
            form_metadata, page_markdown, abstract_pages
        )
    qa_report = build_qa_report(page_confidence, field_evidence, corrections)

    missing_required = [
        field
        for field in ("title", "authors")
        if not form_metadata[field]
    ]
    warnings: list[str] = []
    if missing_required:
        warnings.append(
            "Required form values missing: " + ", ".join(missing_required)
        )
    for field, item in field_evidence.items():
        if field in {"abstract", "note"}:
            continue
        if item.get("status") in {
            "not_found_in_ocr",
            "partial_or_not_found",
            "not_independently_verified",
        }:
            warnings.append(f"Review {field}: {item.get('status')}")

    markdown_pages = [
        f"## Page {index + 1}\n\n{page_markdown[index]}\n"
        for index in range(page_count)
    ]
    text_path = output_dir / "text.md"
    text_path.write_text(
        "# Mistral Document AI OCR result\n\n"
        f"- Source: `{args.pdf.as_posix()}`\n"
        f"- Pages: {page_count}\n"
        f"- Model requested: `{args.model}`\n"
        f"- SHA-256: `{sha256}`\n\n"
        + "\n---\n\n".join(markdown_pages),
        encoding="utf-8",
    )

    write_json(output_dir / "form_metadata.json", form_metadata)
    write_json(output_dir / "corrections.json", corrections)
    write_json(output_dir / "qa_report.json", qa_report)
    write_json(
        output_dir / "metadata.json",
        {
            "form": form_metadata,
            "field_evidence": field_evidence,
            "warnings": warnings,
            "verification_status": (
                "Fields marked verified_in_ocr were matched against OCR text. DOI, "
                "year, pages, publisher, and identifiers should still be checked "
                "against authoritative sources before database ingestion."
            ),
            "annotation_source_pages": [1, min(METADATA_PAGE_COUNT, page_count)],
            "raw_annotation_response": annotation_raw_file,
        },
    )
    write_json(
        output_dir / "manifest.json",
        {
            "source": args.pdf.as_posix(),
            "source_sha256": sha256,
            "page_count": page_count,
            "ocr_chunks": [[page + 1 for page in chunk] for chunk in chunks],
            "annotation_source_pages": [1, min(METADATA_PAGE_COUNT, page_count)],
            "api_call_count": len(chunks),
            "raw_responses_reused_for_rebuild": args.reuse_raw,
            "model_requested": args.model,
            "text_file": text_path.name,
            "form_metadata_file": "form_metadata.json",
            "metadata_evidence_file": "metadata.json",
            "corrections_file": "corrections.json",
            "qa_report_file": "qa_report.json",
            "usage_by_call": usage_records,
            "page_confidence": page_confidence,
        },
    )
    print(f"Done. OCR text: {text_path}")
    print(f"Form metadata: {output_dir / 'form_metadata.json'}")
    if warnings:
        print("Warnings: " + "; ".join(warnings), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
