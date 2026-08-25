#!/usr/bin/env python3
"""Batch-extract literature-form metadata and export a flat UTF-8 CSV."""

from __future__ import annotations

import argparse
import base64
import csv
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import httpx
from pypdf import PdfReader, PdfWriter

from mistral_document_ai_experiment import (
    DEFAULT_ENV_FILE,
    build_field_evidence,
    correct_cover_title,
    drop_unverified_optional_claims,
    load_env_file,
    markdown_with_inline_tables,
    normalize_metadata,
    parse_annotation,
    process_ocr,
    write_json,
)


DEFAULT_METADATA_PAGES = 8
MAX_METADATA_PAGES = 8
LOCAL_TEXT_SCAN_PAGES = 30

CSV_COLUMNS = [
    "source_file",
    "source_total_pages",
    "metadata_pages_processed",
    "metadata_source_pages",
    "title",
    "authors",
    "venue_or_publisher",
    "document_type",
    "year",
    "pages",
    "doi",
    "abstract",
    "note",
    "average_page_confidence",
    "review_status",
    "review_notes",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def metadata_page_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed > MAX_METADATA_PAGES:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_METADATA_PAGES}"
        )
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-extract form-ready metadata from a PDF folder with Mistral "
            "Document AI and write UTF-8 CSV/JSON output."
        )
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        type=Path,
        help="Optional explicit PDF paths (alternative to --input-dir).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Folder containing the PDFs to process.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --input-dir and all its subfolders.",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Process only the first N PDFs after deterministic filename sorting.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder (default: result/<input-folder>_metadata).",
    )
    parser.add_argument(
        "--metadata-pages",
        type=metadata_page_count,
        default=DEFAULT_METADATA_PAGES,
        help=(
            "Pages sent per PDF for metadata extraction; cover/title/CIP pages are "
            f"selected automatically (1-{MAX_METADATA_PAGES}, default: "
            f"{DEFAULT_METADATA_PAGES})."
        ),
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to the .env file containing MISTRAL_API_KEY (default: .env).",
    )
    parser.add_argument("--model", default="mistral-ocr-latest")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Rebuild JSON/CSV from existing raw response files without API calls.",
    )
    parser.add_argument(
        "--refresh-index",
        action="append",
        type=positive_int,
        default=[],
        help=(
            "With --reuse-raw, rerun one 1-based input index through the API. "
            "Repeat this option to refresh multiple PDFs."
        ),
    )
    args = parser.parse_args(argv)
    if args.input_dir is not None and args.pdfs:
        parser.error("use either --input-dir or positional PDF paths, not both")
    if args.input_dir is None and not args.pdfs:
        parser.error("provide --input-dir or at least one PDF path")
    if args.recursive and args.input_dir is None:
        parser.error("--recursive requires --input-dir")
    if args.refresh_index and not args.reuse_raw:
        parser.error("--refresh-index requires --reuse-raw")
    return args


def collect_pdf_paths(
    args: argparse.Namespace,
) -> tuple[list[Path], Path | None]:
    input_root: Path | None = None
    if args.input_dir is not None:
        input_root = args.input_dir.expanduser().resolve()
        if not input_root.is_dir():
            raise NotADirectoryError(input_root)
        candidates = input_root.rglob("*") if args.recursive else input_root.iterdir()
        paths = [
            path.absolute()
            for path in candidates
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]
        paths.sort(
            key=lambda path: path.relative_to(input_root).as_posix().casefold()
        )
    else:
        paths = [path.expanduser().resolve() for path in args.pdfs]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() != ".pdf":
                raise ValueError(f"Not a PDF file: {path}")

    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        location = input_root if input_root is not None else "the supplied paths"
        raise FileNotFoundError(f"No PDF files found in {location}")
    return paths, input_root


def source_label(path: Path, input_root: Path | None) -> str:
    if input_root is not None:
        return path.relative_to(input_root).as_posix()
    return path.name


def default_output_dir(input_root: Path | None) -> Path:
    suffix = f"{input_root.name}_metadata" if input_root else "pdf_metadata_batch"
    return Path("result") / suffix


def metadata_page_score(text: str) -> int:
    """Rank pages likely to contain publication facts using local embedded text."""
    if not text:
        return 0
    score = 0
    weighted_patterns = (
        (r"\bISBN(?:-1[03])?\b", 10),
        (r"图书在版编目|中国版本图书馆CIP数据", 10),
        (r"catalog(?:ing|uing)[ -]in[ -]publication", 10),
        (r"copyright|版权所有|版次|印次", 7),
        (r"出版社|出版发行|published\s+by|publisher", 6),
        (r"\bdoi\s*:|https?://doi\.org/|\b10\.\d{4,9}/", 6),
    )
    for pattern, weight in weighted_patterns:
        if re.search(pattern, text, flags=re.I):
            score += weight
    return score


def source_page_indexes(reader: PdfReader, limit: int) -> list[int]:
    """Choose cover/title pages plus likely copyright/CIP pages."""
    total_pages = len(reader.pages)
    target = min(limit, total_pages)
    selected = list(range(min(3, target)))

    ranked: list[tuple[int, int]] = []
    for page_index in range(min(total_pages, LOCAL_TEXT_SCAN_PAGES)):
        if page_index in selected:
            continue
        try:
            text = reader.pages[page_index].extract_text() or ""
        except Exception:
            text = ""
        score = metadata_page_score(text)
        if score:
            ranked.append((score, page_index))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    for _, page_index in ranked:
        if len(selected) >= target:
            break
        selected.append(page_index)

    for page_index in range(total_pages):
        if len(selected) >= target:
            break
        if page_index not in selected:
            selected.append(page_index)
    return sorted(selected)


def pdf_subset_bytes(
    reader: PdfReader, selected_source_indexes: list[int]
) -> bytes:
    writer = PdfWriter()
    for page_index in selected_source_indexes:
        writer.add_page(reader.pages[page_index])
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def remap_evidence_pages(
    evidence: dict[str, Any], selected_source_indexes: list[int]
) -> dict[str, Any]:
    def remap(pages: list[int]) -> list[int]:
        return [selected_source_indexes[page - 1] + 1 for page in pages]

    for item in evidence.values():
        if isinstance(item.get("pages"), list):
            item["pages"] = remap(item["pages"])
        for nested_key in ("pages_by_author", "pages_by_component"):
            nested = item.get(nested_key)
            if isinstance(nested, dict):
                item[nested_key] = {
                    key: remap(pages) for key, pages in nested.items()
                }
    return evidence


def average_page_confidence(response_pages: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for page in response_pages:
        score = (page.get("confidence_scores") or {}).get(
            "average_page_confidence_score"
        )
        if isinstance(score, (int, float)):
            values.append(float(score))
    return sum(values) / len(values) if values else None


def build_review_notes(
    form: dict[str, Any], evidence: dict[str, Any]
) -> list[str]:
    notes: list[str] = []
    if not form["title"]:
        notes.append("missing_title")
    if not form["authors"]:
        notes.append("missing_authors")
    for field in (
        "title",
        "authors",
        "venue_or_publisher",
        "document_type",
        "year",
        "pages",
        "doi",
        "abstract",
        "note",
    ):
        value = form[field]
        if value is None or value == [] or value == "":
            continue
        status = evidence[field]["status"]
        if status not in {"verified_in_ocr", "transcribed_from_printed_section"}:
            notes.append(f"review_{field}:{status}")
    return notes


def csv_row(record: dict[str, Any]) -> dict[str, Any]:
    form = record["form"]
    confidence = record["average_page_confidence"]
    return {
        "source_file": record["source_file"],
        "source_total_pages": record["source_total_pages"],
        "metadata_pages_processed": record["metadata_pages_processed"],
        "metadata_source_pages": ",".join(
            str(page) for page in record["metadata_source_pages"]
        ),
        "title": form["title"] or "",
        "authors": ", ".join(form["authors"]),
        "venue_or_publisher": form["venue_or_publisher"] or "",
        "document_type": form["document_type"],
        "year": form["year"] if form["year"] is not None else "",
        "pages": form["pages"] or "",
        "doi": form["doi"] or "",
        "abstract": " ".join((form["abstract"] or "").split()),
        "note": " ".join((form["note"] or "").split()),
        "average_page_confidence": (
            round(confidence, 6) if confidence is not None else ""
        ),
        "review_status": record["review_status"],
        "review_notes": "; ".join(record["review_notes"]),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_row(record))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pdf_paths, input_root = collect_pdf_paths(args)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    refresh_indexes = set(args.refresh_index)
    invalid_refresh_indexes = sorted(
        index for index in refresh_indexes if index > len(pdf_paths)
    )
    if invalid_refresh_indexes:
        print(
            "--refresh-index is outside the selected input range: "
            + ", ".join(str(index) for index in invalid_refresh_indexes),
            file=sys.stderr,
        )
        return 2

    load_env_file(args.env_file)
    api_key = os.environ.get("MISTRAL_API_KEY")
    if (not args.reuse_raw or refresh_indexes) and not api_key:
        print(
            f"MISTRAL_API_KEY is not set. Add it to {args.env_file} or export it "
            "in the shell.",
            file=sys.stderr,
        )
        return 2

    output_dir: Path = args.output_dir or default_output_dir(input_root)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    print(f"Selected PDFs: {len(pdf_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata page limit per PDF: {args.metadata_pages}")

    client: httpx.Client | None = None
    if not args.reuse_raw or refresh_indexes:
        client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=args.timeout
        )
    try:
        for position, path in enumerate(pdf_paths, start=1):
            label = source_label(path, input_root)
            raw_path = raw_dir / f"{position:03d}.json"
            print(f"[{position}/{len(pdf_paths)}] {label}")
            try:
                use_raw = args.reuse_raw and position not in refresh_indexes
                if use_raw:
                    if not raw_path.is_file():
                        raise FileNotFoundError(raw_path)
                    stored_response = json.loads(raw_path.read_text(encoding="utf-8"))
                    source_info = stored_response.get("_batch_source") or {}
                    stored_label = source_info.get("source_file")
                    if stored_label and stored_label != label:
                        raise RuntimeError(
                            f"Raw response belongs to {stored_label!r}, not {label!r}"
                        )
                    response = {
                        key: value
                        for key, value in stored_response.items()
                        if key != "_batch_source"
                    }
                    total_pages = int(source_info["source_total_pages"])
                    selected_pages = int(source_info["metadata_pages_processed"])
                    selected_source_indexes = source_info.get("source_page_indexes")
                    if selected_source_indexes is None:
                        selected_source_indexes = list(range(selected_pages))
                    selected_source_indexes = [
                        int(index) for index in selected_source_indexes
                    ]
                else:
                    reader = PdfReader(path)
                    total_pages = len(reader.pages)
                    if total_pages == 0:
                        raise ValueError("PDF has no pages")
                    selected_source_indexes = source_page_indexes(
                        reader, args.metadata_pages
                    )
                    selected_pages = len(selected_source_indexes)
                    subset = pdf_subset_bytes(reader, selected_source_indexes)
                    data_url = "data:application/pdf;base64," + base64.b64encode(
                        subset
                    ).decode("ascii")
                    if client is None:
                        raise RuntimeError("HTTP client was not initialized")
                    response = process_ocr(
                        client,
                        model=args.model,
                        pdf_data_url=data_url,
                        page_indexes=list(range(selected_pages)),
                        extract_metadata=True,
                    )
                    stored_response = {
                        **response,
                        "_batch_source": {
                            "source_file": label,
                            "source_total_pages": total_pages,
                            "metadata_pages_processed": selected_pages,
                            "source_page_indexes": selected_source_indexes,
                        },
                    }
                    write_json(raw_path, stored_response)

                if len(selected_source_indexes) != selected_pages:
                    raise RuntimeError("Raw metadata page mapping is inconsistent")
                if any(
                    index < 0 or index >= total_pages
                    for index in selected_source_indexes
                ):
                    raise RuntimeError("Raw metadata page mapping is outside the PDF")

                response_pages = response.get("pages") or []
                if len(response_pages) != selected_pages:
                    raise RuntimeError(
                        f"Expected {selected_pages} pages, got {len(response_pages)}"
                    )
                page_markdown = {
                    index: markdown_with_inline_tables(page).strip()
                    for index, page in enumerate(response_pages)
                }
                annotation = parse_annotation(response["document_annotation"])
                form, abstract_pages = normalize_metadata(
                    annotation,
                    page_markdown,
                    metadata_page_count=selected_pages,
                )
                evidence = build_field_evidence(form, page_markdown, abstract_pages)
                evidence_adjustments = drop_unverified_optional_claims(form, evidence)
                if evidence_adjustments:
                    evidence = build_field_evidence(
                        form, page_markdown, abstract_pages
                    )
                corrections = correct_cover_title(
                    page_markdown, form["title"], evidence["title"]["pages"]
                )
                if corrections:
                    evidence = build_field_evidence(
                        form, page_markdown, abstract_pages
                    )
                evidence = remap_evidence_pages(evidence, selected_source_indexes)
                review_notes = build_review_notes(form, evidence)
                records.append(
                    {
                        "source_file": label,
                        "source_total_pages": total_pages,
                        "metadata_pages_processed": selected_pages,
                        "metadata_source_pages": [
                            index + 1 for index in selected_source_indexes
                        ],
                        "form": form,
                        "field_evidence": evidence,
                        "evidence_adjustments": evidence_adjustments,
                        "corrections": corrections,
                        "average_page_confidence": average_page_confidence(
                            response_pages
                        ),
                        "review_status": (
                            "verified" if not review_notes else "review_needed"
                        ),
                        "review_notes": review_notes,
                        "raw_response": raw_path.relative_to(output_dir).as_posix(),
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "source_file": label,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        if client is not None:
            client.close()

    write_json(
        output_dir / "batch_metadata.json",
        {
            "input_directory": input_root.as_posix() if input_root else None,
            "model_requested": args.model,
            "metadata_page_limit": args.metadata_pages,
            "record_count": len(records),
            "failure_count": len(failures),
            "records": records,
            "failures": failures,
        },
    )
    csv_path = output_dir / "literature_metadata.csv"
    write_csv(csv_path, records)
    print(f"CSV: {csv_path}")
    print(f"Records: {len(records)}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
