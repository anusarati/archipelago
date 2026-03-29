from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import tarfile
import time
import hashlib
import gzip
import warnings
import zipfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np

ARCHIPELAGO_DIR = Path(__file__).resolve().parents[2]
HSN_PROJECT_DIR = ARCHIPELAGO_DIR / "hierarchical-semantic-navigation"

if str(HSN_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(HSN_PROJECT_DIR))

HSN_INDEX_REL_PATH = "hsn/current_world.json"
HSN_CACHE_ROOT = Path(
    os.environ.get("HSN_CACHE_ROOT", Path.home() / ".cache" / "archipelago" / "hsn")
)
HSN_EMBEDDING_MODEL = os.environ.get(
    "HSN_EMBEDDING_MODEL", "openai/text-embedding-3-small"
)
HSN_EMBEDDING_BATCH_SIZE = int(os.environ.get("HSN_EMBEDDING_BATCH_SIZE", "1024"))
HSN_EMBEDDING_API_KEY = os.environ.get("HSN_EMBEDDING_API_KEY") or os.environ.get(
    "OPENAI_API_KEY"
)
HSN_EMBEDDING_BASE_URL = (
    os.environ.get("HSN_EMBEDDING_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or "https://api.openai.com/v1"
).rstrip("/")
HSN_EMBEDDING_TIMEOUT_SECONDS = float(
    os.environ.get("HSN_EMBEDDING_TIMEOUT_SECONDS", "180")
)
_HSN_MAX_TEXT_CHARS_LEGACY = int(os.environ.get("HSN_MAX_TEXT_CHARS", "131072"))
HSN_MAX_EXTRACTED_TEXT_CHARS = int(
    os.environ.get("HSN_MAX_EXTRACTED_TEXT_CHARS", str(_HSN_MAX_TEXT_CHARS_LEGACY))
)
HSN_MAX_EMBEDDING_TEXT_CHARS = int(
    os.environ.get("HSN_MAX_EMBEDDING_TEXT_CHARS", 65535)
)
HSN_MAX_FILES = int(os.environ.get("HSN_MAX_FILES", "5000"))
HSN_EXTRACTION_CACHE_VERSION = 3
HSN_MAX_GARBLED_WARNINGS = int(os.environ.get("HSN_MAX_GARBLED_WARNINGS", "20"))
HSN_PDF_OCR_DPI = int(os.environ.get("HSN_PDF_OCR_DPI", "150"))
HSN_PDF_OCR_MAX_PAGES = int(os.environ.get("HSN_PDF_OCR_MAX_PAGES", "128"))
HSN_PDF_OCR_LANG = os.environ.get("HSN_PDF_OCR_LANG", "eng")
HSN_PDF_OCR_CONFIG = os.environ.get("HSN_PDF_OCR_CONFIG", "--psm 6")
HSN_PDF_OCR_PAGE_TIMEOUT_SECONDS = int(
    os.environ.get("HSN_PDF_OCR_PAGE_TIMEOUT_SECONDS", "20")
)
HSN_PDF_OCR_THREAD_COUNT = int(os.environ.get("HSN_PDF_OCR_THREAD_COUNT", "2"))

DOCUMENT_EXTENSIONS = {
    "txt",
    "md",
    "rst",
    "json",
    "yaml",
    "yml",
    "csv",
    "tsv",
    "xml",
    "html",
    "htm",
    "pdf",
    "doc",
    "docx",
    "rtf",
    "odt",
    "xls",
    "xlsx",
    "ods",
    "ppt",
    "pptx",
    "odp",
    "eml",
    "msg",
    "log",
}

TEXT_EXTENSIONS = {
    "txt",
    "md",
    "rst",
    "json",
    "yaml",
    "yml",
    "csv",
    "tsv",
    "xml",
    "html",
    "htm",
    "log",
}


@dataclass
class PreparedHSN:
    index_path: Path
    index_data: dict[str, Any]
    initial_message: str


def hsn_enabled() -> bool:
    raw = os.environ.get("USE_HSN_FILESYSTEM", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _configure_document_extraction_logging() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Unknown extension is not supported and will be removed",
        module=r"openpyxl\.worksheet\._reader",
    )
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfminer.pdffont").setLevel(logging.ERROR)


def _build_converter() -> Any:
    _configure_document_extraction_logging()
    try:
        from markitdown import MarkItDown  # import locally for optional dependency
    except Exception as exc:
        raise RuntimeError(
            "HSN mode requires markitdown for non-PDF document extraction. "
            "Install dependencies with `cd agents && uv sync`."
        ) from exc
    return MarkItDown()


def _build_hsn_builder() -> Any:
    try:
        from hsn.builder import HierarchicalGraphBuilder  # import locally for optional dependency
    except Exception as exc:
        raise RuntimeError(
            "HSN mode requires hierarchical-semantic-navigation + hnswlib. "
            "Install dependencies with `cd agents && uv sync`."
        ) from exc
    return HierarchicalGraphBuilder()


def _normalize_path(path: str) -> str:
    normalized = os.path.normpath("/" + path.lstrip("/"))
    return normalized if normalized != "." else "/"


def _extension(path: str) -> str:
    name = os.path.basename(path).lower()
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1]


def _document_like(path: str) -> bool:
    return _extension(path) in DOCUMENT_EXTENSIONS


def _load_embedding_extra_args() -> dict[str, Any]:
    raw = os.environ.get("HSN_EMBEDDING_EXTRA_ARGS", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid HSN_EMBEDDING_EXTRA_ARGS JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("HSN_EMBEDDING_EXTRA_ARGS must be a JSON object")
    return payload


def _extract_pdf_text_with_ocr(path: str, data: bytes, log: Callable[[str], None]) -> str:
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except Exception as exc:
        raise RuntimeError(
            "HSN PDF OCR requires `pdf2image` and `pytesseract`. Install with `cd agents && uv sync`."
        ) from exc

    if shutil.which("tesseract") is None:
        log(f"  WARNING: PDF OCR skipped for {path}: `tesseract` binary not found")
        return ""
    if shutil.which("pdftoppm") is None and shutil.which("pdftocairo") is None:
        log(
            f"  WARNING: PDF OCR skipped for {path}: Poppler binary (`pdftoppm`/`pdftocairo`) not found"
        )
        return ""

    try:
        pages = convert_from_bytes(
            data,
            dpi=HSN_PDF_OCR_DPI,
            first_page=1,
            last_page=max(1, HSN_PDF_OCR_MAX_PAGES),
            thread_count=max(1, HSN_PDF_OCR_THREAD_COUNT),
        )
    except Exception as exc:
        log(f"  WARNING: PDF OCR failed for {path}: {exc}")
        return ""

    chunks: list[str] = []
    total_chars = 0
    for page_num, page_image in enumerate(pages, start=1):
        try:
            page_text = pytesseract.image_to_string(
                page_image,
                lang=HSN_PDF_OCR_LANG,
                config=HSN_PDF_OCR_CONFIG,
                timeout=max(1, HSN_PDF_OCR_PAGE_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            log(f"  WARNING: PDF OCR page {page_num} failed for {path}: {exc}")
            continue

        if page_text:
            chunks.append(page_text)
            total_chars += len(page_text)
            if total_chars >= HSN_MAX_EXTRACTED_TEXT_CHARS:
                break

    text = "\n".join(chunks).strip()
    if not text:
        log(f"  WARNING: PDF OCR produced empty/whitespace text for {path}")
    return text[:HSN_MAX_EXTRACTED_TEXT_CHARS]


def _extract_text(path: str, data: bytes, converter: Any, log: Callable[[str], None]) -> str:
    ext = _extension(path)
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="ignore")[:HSN_MAX_EXTRACTED_TEXT_CHARS]
    if ext == "pdf":
        return _extract_pdf_text_with_ocr(path, data, log)

    try:
        document = converter.convert(io.BytesIO(data))
        text = getattr(document, "text_content", None)
        if not text:
            text = str(document) if document is not None else ""
        if text:
            return text[:HSN_MAX_EXTRACTED_TEXT_CHARS]
    except Exception:
        pass

    return data.decode("latin-1", errors="ignore")[:HSN_MAX_EXTRACTED_TEXT_CHARS]


def _max_non_whitespace_run(text: str) -> int:
    max_run = 0
    current = 0
    for ch in text:
        if ch.isspace():
            if current > max_run:
                max_run = current
            current = 0
        else:
            current += 1
    return max(max_run, current)


def _garbled_text_reasons(text: str) -> list[str]:
    if not text:
        return []

    sample = text[:10000]
    sample_len = len(sample)
    if sample_len == 0:
        return []

    escaped_hex_sequences = len(re.findall(r"\\x[0-9A-Fa-f]{2}", sample))
    control_chars = sum(
        1 for ch in sample if (ord(ch) < 32 and ch not in {"\n", "\r", "\t"})
    )
    alpha_chars = sum(1 for ch in sample if ch.isalpha())
    whitespace_chars = sum(1 for ch in sample if ch.isspace())
    readable_ratio = (alpha_chars + whitespace_chars) / sample_len
    max_token_run = _max_non_whitespace_run(sample)

    reasons: list[str] = []
    if escaped_hex_sequences >= 5:
        reasons.append(f"many escaped hex sequences ({escaped_hex_sequences})")
    if control_chars > 0 and (control_chars / sample_len) > 0.01:
        reasons.append(f"control chars ratio {(control_chars / sample_len):.2%}")
    if sample_len >= 200 and readable_ratio < 0.25:
        reasons.append(f"low readable ratio ({readable_ratio:.2%})")
    if sample_len >= 200 and max_token_run >= 140:
        reasons.append(f"very long no-whitespace run ({max_token_run} chars)")
    return reasons


def _log_garbled_document_warnings(
    docs: list[dict[str, Any]], log: Callable[[str], None], source: str
) -> None:
    warned = 0
    suppressed = 0
    for doc in docs:
        path = doc.get("path")
        text = doc.get("text")
        if not isinstance(path, str) or not isinstance(text, str):
            continue
        reasons = _garbled_text_reasons(text)
        if not reasons:
            continue
        if warned < HSN_MAX_GARBLED_WARNINGS:
            preview = text[:120].encode("unicode_escape", errors="ignore").decode("ascii")
            log(
                "  WARNING: HSN extracted text may be garbled "
                f"({source}) for {path}: {', '.join(reasons)} | preview={preview}"
            )
            warned += 1
        else:
            suppressed += 1
    if suppressed:
        log(
            f"  WARNING: HSN garbled-text warnings suppressed for {suppressed} additional files"
        )


def _collect_documents(world_zip: Path, log: Callable[[str], None]) -> list[dict[str, Any]]:
    converter = _build_converter()
    docs: list[dict[str, Any]] = []
    with zipfile.ZipFile(world_zip, "r") as zf:
        names = sorted(
            n for n in zf.namelist() if n.startswith("filesystem/") and not n.endswith("/")
        )
        candidates: list[tuple[str, str]] = []
        for name in names:
            rel_path = _normalize_path(name[len("filesystem/") :])
            if not _document_like(rel_path):
                continue
            candidates.append((name, rel_path))
            if len(candidates) >= HSN_MAX_FILES:
                break

        total_candidates = len(candidates)
        log(f"  HSN: extracting text from {total_candidates} document-like files")
        start_time = time.time()
        for i, (name, rel_path) in enumerate(candidates, start=1):
            data = zf.read(name)
            text = _extract_text(rel_path, data, converter, log)
            docs.append(
                {
                    "id": str(len(docs) + 1),
                    "path": rel_path,
                    "size": len(data),
                    "text": text,
                }
            )
            if i == total_candidates or i % 10 == 0:
                elapsed = time.time() - start_time
                log(f"  HSN: extracted {i}/{total_candidates} files ({elapsed:.1f}s elapsed)")

    _log_garbled_document_warnings(docs, log, source="fresh extraction")
    log(f"  HSN: collected {len(docs)} document-like files")
    return docs


def _embedding_input_digest(texts: list[str]) -> str:
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8", errors="ignore"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _partial_embedding_cache_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    meta_path = cache_dir / "embeddings_partial.meta.json"
    indices_path = cache_dir / "embeddings_partial.indices.json"
    vectors_path = cache_dir / "embeddings_partial.npy"
    return meta_path, indices_path, vectors_path


def _load_partial_embeddings(
    cache_dir: Path,
    cache_key: dict[str, Any],
    log: Callable[[str], None],
) -> dict[int, list[float]]:
    meta_path, indices_path, vectors_path = _partial_embedding_cache_paths(cache_dir)
    if not meta_path.exists() or not indices_path.exists() or not vectors_path.exists():
        return {}

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            return {}
        if meta.get("version") != 1:
            return {}
        if meta.get("cache_key") != cache_key:
            return {}

        with open(indices_path) as f:
            indices_raw = json.load(f)
        if not isinstance(indices_raw, list):
            return {}
        indices = [int(v) for v in indices_raw]

        vectors = np.load(vectors_path)
        if vectors.ndim != 2 or vectors.shape[0] != len(indices):
            return {}

        out: dict[int, list[float]] = {}
        for row, idx in enumerate(indices):
            out[idx] = [float(v) for v in vectors[row].tolist()]
        if out:
            log(
                f"  HSN: partial embedding cache hit ({len(out)} / {cache_key.get('input_count', '?')})"
            )
        return out
    except Exception:
        return {}


def _save_partial_embeddings(
    cache_dir: Path,
    cache_key: dict[str, Any],
    vectors_by_index: dict[int, list[float]],
) -> None:
    if not vectors_by_index:
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    sorted_indices = sorted(vectors_by_index)
    matrix = np.array([vectors_by_index[idx] for idx in sorted_indices], dtype=np.float32)
    meta_path, indices_path, vectors_path = _partial_embedding_cache_paths(cache_dir)

    np.save(vectors_path, matrix)
    with open(indices_path, "w") as f:
        json.dump(sorted_indices, f)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "version": 1,
                "cache_key": cache_key,
                "count": len(sorted_indices),
                "created_at": int(time.time()),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def _request_embeddings(
    client: Any, batch: list[str], extra_args: dict[str, Any]
) -> list[list[float]]:
    payload: dict[str, Any] = {
        "model": HSN_EMBEDDING_MODEL,
        "input": batch,
        **extra_args,
    }
    response = client.embeddings.create(**payload)
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != len(batch):
        raise RuntimeError("Unexpected embedding response shape from embedding endpoint")

    vectors: list[list[float]] = []
    for item in data:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError("Missing embedding vector in embedding response")
        vectors.append([float(v) for v in embedding])
    return vectors


def _bisect_failed_embeddings(
    *,
    client: Any,
    texts: list[str],
    indices: list[int],
    labels: list[str],
    extra_args: dict[str, Any],
    log: Callable[[str], None],
) -> tuple[dict[int, list[float]], list[tuple[int, str]]]:
    recovered: dict[int, list[float]] = {}
    failures: list[tuple[int, str]] = []

    def recurse(sub_indices: list[int]) -> None:
        if not sub_indices:
            return
        sub_texts = [texts[idx] for idx in sub_indices]
        try:
            vectors = _request_embeddings(client, sub_texts, extra_args)
            for idx, vector in zip(sub_indices, vectors):
                recovered[idx] = vector
            return
        except Exception as exc:
            if len(sub_indices) == 1:
                idx = sub_indices[0]
                label = labels[idx] if idx < len(labels) else f"index:{idx}"
                err = str(exc)
                log(
                    "  HSN: isolated failing embedding input "
                    f"idx={idx + 1}/{len(labels)} path={label} chars={len(texts[idx])} error={err}"
                )
                failures.append((idx, err))
                return

        mid = len(sub_indices) // 2
        recurse(sub_indices[:mid])
        recurse(sub_indices[mid:])

    recurse(indices)
    return recovered, failures


def _extract_embeddings(
    texts: list[str],
    labels: list[str],
    cache_dir: Path,
    log: Callable[[str], None],
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if HSN_EMBEDDING_BATCH_SIZE <= 0:
        raise ValueError("HSN_EMBEDDING_BATCH_SIZE must be >= 1")
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "HSN embeddings require the `openai` package. Install dependencies with `cd agents && uv sync`."
        ) from exc

    extra_args = _load_embedding_extra_args()
    total = len(texts)
    num_batches = (total + HSN_EMBEDDING_BATCH_SIZE - 1) // HSN_EMBEDDING_BATCH_SIZE
    cache_key = {
        "model": HSN_EMBEDDING_MODEL,
        "base_url": HSN_EMBEDDING_BASE_URL,
        "embedding_input_max_chars": HSN_MAX_EMBEDDING_TEXT_CHARS,
        "extra_args": extra_args,
        "input_digest": _embedding_input_digest(texts),
        "input_count": total,
    }
    vectors_by_index = _load_partial_embeddings(cache_dir, cache_key, log)

    client = OpenAI(
        api_key=HSN_EMBEDDING_API_KEY,
        base_url=HSN_EMBEDDING_BASE_URL,
        timeout=HSN_EMBEDDING_TIMEOUT_SECONDS,
    )

    for batch_idx, start in enumerate(range(0, total, HSN_EMBEDDING_BATCH_SIZE), start=1):
        end = min(total, start + HSN_EMBEDDING_BATCH_SIZE)
        batch_indices = list(range(start, end))
        missing_indices = [idx for idx in batch_indices if idx not in vectors_by_index]
        if not missing_indices:
            log(
                f"  HSN: embedding batch {batch_idx}/{num_batches} "
                f"({start + 1}-{end}/{total}) skipped (cache hit)"
            )
            continue

        batch = [texts[idx] for idx in missing_indices]
        log(f"  HSN: embedding batch {batch_idx}/{num_batches} ({start + 1}-{end}/{total})")

        try:
            batch_vectors = _request_embeddings(client, batch, extra_args)
        except Exception as exc:
            log(
                "  WARNING: embedding batch request failed; "
                "running binary search to isolate failing input(s)"
            )
            recovered, failures = _bisect_failed_embeddings(
                client=client,
                texts=texts,
                indices=missing_indices,
                labels=labels,
                extra_args=extra_args,
                log=log,
            )
            vectors_by_index.update(recovered)
            _save_partial_embeddings(cache_dir, cache_key, vectors_by_index)

            if failures:
                details: list[str] = []
                for idx, err in failures:
                    label = labels[idx] if idx < len(labels) else f"index:{idx}"
                    details.append(
                        f"idx={idx + 1}/{total} path={label} chars={len(texts[idx])} error={err}"
                    )
                raise RuntimeError(
                    "Embedding failures detected after binary search; aborting run.\n"
                    + "\n".join(details)
                ) from exc
            raise RuntimeError(
                "Embedding batch failed but binary search did not isolate failing inputs; aborting run."
            ) from exc

        for idx, vector in zip(missing_indices, batch_vectors):
            vectors_by_index[idx] = vector
        _save_partial_embeddings(cache_dir, cache_key, vectors_by_index)

    missing = [idx for idx in range(total) if idx not in vectors_by_index]
    if missing:
        raise RuntimeError(
            "Embedding run incomplete: missing vectors for indices "
            + ", ".join(str(idx + 1) for idx in missing[:20])
        )

    ordered_vectors = [vectors_by_index[idx] for idx in range(total)]
    return np.array(ordered_vectors, dtype=np.float32)


def _build_embedding_input(path: str, text: str) -> str:
    # Some embedding backends enforce a strict "less than N chars" limit.
    max_total_chars = max(1, HSN_MAX_EMBEDDING_TEXT_CHARS - 1)
    prefix = f"path: {path}\n"
    if len(prefix) >= max_total_chars:
        return prefix[:max_total_chars]
    return prefix + text[: max_total_chars - len(prefix)]


def _sanitize_model_name(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    return slug.strip("_") or "default_model"


def _cache_root_for_world(world_id: str) -> Path:
    return HSN_CACHE_ROOT / world_id


def _cache_dir_for_world(world_id: str) -> Path:
    return _cache_root_for_world(world_id) / _sanitize_model_name(HSN_EMBEDDING_MODEL)


def _extraction_cache_dir_for_world(world_id: str) -> Path:
    return _cache_root_for_world(world_id) / "_extraction"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _markitdown_version() -> str:
    try:
        from importlib.metadata import version

        return version("markitdown")
    except Exception:
        return "unknown"


def _load_cached_extraction(
    extraction_cache_dir: Path, cache_key: dict[str, Any]
) -> list[dict[str, Any]] | None:
    docs_path = extraction_cache_dir / "documents_with_text.json.gz"
    meta_path = extraction_cache_dir / "meta.json"
    if not docs_path.exists() or not meta_path.exists():
        return None

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            return None
        if meta.get("version") != HSN_EXTRACTION_CACHE_VERSION:
            return None
        if meta.get("cache_key") != cache_key:
            return None
        with gzip.open(docs_path, "rt", encoding="utf-8") as f:
            docs = json.load(f)
        if not isinstance(docs, list):
            return None
        return docs
    except Exception:
        return None


def _save_cached_extraction(
    extraction_cache_dir: Path,
    docs: list[dict[str, Any]],
    cache_key: dict[str, Any],
) -> None:
    extraction_cache_dir.mkdir(parents=True, exist_ok=True)
    docs_path = extraction_cache_dir / "documents_with_text.json.gz"
    meta_path = extraction_cache_dir / "meta.json"
    with gzip.open(docs_path, "wt", encoding="utf-8") as f:
        json.dump(docs, f)
    with open(meta_path, "w") as f:
        json.dump(
            {
                "version": HSN_EXTRACTION_CACHE_VERSION,
                "cache_key": cache_key,
                "doc_count": len(docs),
                "created_at": int(time.time()),
            },
            f,
            indent=2,
            sort_keys=True,
        )


def _load_or_collect_documents(
    world_id: str,
    world_zip: Path,
    log: Callable[[str], None],
) -> list[dict[str, Any]]:
    log("  HSN: computing world fingerprint for extraction cache...")
    world_sha256 = _sha256_file(world_zip)
    cache_key = {
        "world_sha256": world_sha256,
        "max_extracted_text_chars": HSN_MAX_EXTRACTED_TEXT_CHARS,
        "max_files": HSN_MAX_FILES,
        "markitdown_version": _markitdown_version(),
        "pdf_extraction_method": "pdf2image+pytesseract",
        "pdf_ocr_dpi": HSN_PDF_OCR_DPI,
        "pdf_ocr_max_pages": HSN_PDF_OCR_MAX_PAGES,
        "pdf_ocr_lang": HSN_PDF_OCR_LANG,
        "pdf_ocr_config": HSN_PDF_OCR_CONFIG,
        "pdf_ocr_page_timeout_seconds": HSN_PDF_OCR_PAGE_TIMEOUT_SECONDS,
        "pdf_ocr_thread_count": HSN_PDF_OCR_THREAD_COUNT,
    }

    extraction_cache_dir = _extraction_cache_dir_for_world(world_id)
    cached_docs = _load_cached_extraction(extraction_cache_dir, cache_key)
    if cached_docs is not None:
        log(
            f"  HSN: extraction cache hit: {extraction_cache_dir / 'documents_with_text.json.gz'} "
            f"({len(cached_docs)} files)"
        )
        _log_garbled_document_warnings(cached_docs, log, source="cache")
        return cached_docs

    docs = _collect_documents(world_zip, log)
    _save_cached_extraction(extraction_cache_dir, docs, cache_key)
    log(
        f"  HSN: extraction cache saved: {extraction_cache_dir / 'documents_with_text.json.gz'}"
    )
    return docs


def _graph_to_index(graph: Any, docs: list[dict[str, Any]]) -> dict[str, Any]:
    id_to_path = {doc["id"]: doc["path"] for doc in docs}

    sentinel = -1
    node_to_file_id: dict[int, int] = {}
    for node_id, attrs in graph.nodes(data=True):
        if node_id == sentinel:
            continue
        doc_id = attrs.get("doc_id")
        if doc_id is None:
            continue
        node_to_file_id[int(node_id)] = int(doc_id)

    parents: dict[str, int | None] = {}
    children: dict[str, list[int]] = {}

    for node_id, file_id in node_to_file_id.items():
        predecessors = list(graph.predecessors(node_id))
        parent_file_id: int | None = None
        if predecessors:
            parent_node = int(predecessors[0])
            if parent_node != sentinel:
                parent_file_id = node_to_file_id.get(parent_node)
        parents[str(file_id)] = parent_file_id

    for child_id_str, parent_id in parents.items():
        if parent_id is None:
            continue
        children.setdefault(str(parent_id), []).append(int(child_id_str))

    subtree: dict[str, int] = {}

    def subtree_size(node_id: int) -> int:
        key = str(node_id)
        if key in subtree:
            return subtree[key]
        total = 1
        for child in children.get(key, []):
            total += subtree_size(child)
        subtree[key] = total
        return total

    for doc in docs:
        subtree_size(int(doc["id"]))

    for parent_id, child_ids in children.items():
        child_ids.sort(key=lambda cid: (-subtree.get(str(cid), 1), id_to_path[str(cid)]))

    roots = [int(doc["id"]) for doc in docs if parents.get(str(doc["id"])) is None]
    roots.sort(key=lambda rid: (-subtree.get(str(rid), 1), id_to_path[str(rid)]))

    paths: dict[str, list[int]] = {}

    def node_path(node_id: int) -> list[int]:
        key = str(node_id)
        cached = paths.get(key)
        if cached is not None:
            return cached
        parent_id = parents.get(key)
        if parent_id is None:
            path = [node_id]
        else:
            path = [*node_path(parent_id), node_id]
        paths[key] = path
        return path

    for doc in docs:
        node_path(int(doc["id"]))

    return {
        "version": 1,
        "created_at": int(time.time()),
        "file_count": len(docs),
        "embedding_model": HSN_EMBEDDING_MODEL,
        "id_to_path": id_to_path,
        "path_to_id": {path: int(doc_id) for doc_id, path in id_to_path.items()},
        "parents": parents,
        "children": children,
        "paths": paths,
        "subtree_size": subtree,
        "roots": roots,
    }


def _build_hsn_index(
    world_id: str, world_zip: Path, cache_dir: Path, log: Callable[[str], None]
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    collect_started = time.time()
    docs = _load_or_collect_documents(world_id, world_zip, log)
    log(f"  HSN: text extraction completed in {time.time() - collect_started:.1f}s")
    if not docs:
        return (
            {
                "version": 1,
                "created_at": int(time.time()),
                "world_id": world_id,
                "file_count": 0,
                "embedding_model": HSN_EMBEDDING_MODEL,
                "id_to_path": {},
                "path_to_id": {},
                "parents": {},
                "children": {},
                "paths": {},
                "subtree_size": {},
                "roots": [],
            },
            np.zeros((0, 0), dtype=np.float32),
            docs,
        )

    embed_inputs = [_build_embedding_input(doc["path"], doc["text"]) for doc in docs]
    embed_started = time.time()
    embeddings = _extract_embeddings(
        texts=embed_inputs,
        labels=[doc["path"] for doc in docs],
        cache_dir=cache_dir,
        log=log,
    )
    log(f"  HSN: embedding completed in {time.time() - embed_started:.1f}s")
    builder = _build_hsn_builder()
    graph = builder.build(embeddings, [doc["id"] for doc in docs])
    index = _graph_to_index(graph, docs)
    index["world_id"] = world_id
    index["embedding_input_max_chars"] = HSN_MAX_EMBEDDING_TEXT_CHARS
    index["embedding_extra_args"] = _load_embedding_extra_args()
    return index, embeddings, docs


def _index_cache_compatible(index_data: dict[str, Any]) -> bool:
    expected_extra_args = _load_embedding_extra_args()
    if index_data.get("embedding_model") != HSN_EMBEDDING_MODEL:
        return False
    if index_data.get("embedding_input_max_chars") != HSN_MAX_EMBEDDING_TEXT_CHARS:
        return False
    if index_data.get("embedding_extra_args") != expected_extra_args:
        return False
    return True


def _expand_hsn_nodes_from_data(
    index_data: dict[str, Any],
    start_ids: list[int],
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Expand HSN nodes directly from index_data without reading from disk.

    Returns a list of dicts with keys:
        node_id, path, parent_id, depth, listed_children, total_children
    """
    children_map = index_data.get("children", {})
    subtree_size_map = index_data.get("subtree_size", {})
    paths_map = index_data.get("paths", {})
    id_to_path = index_data.get("id_to_path", {})

    if (
        not isinstance(children_map, dict)
        or not isinstance(subtree_size_map, dict)
        or not isinstance(paths_map, dict)
        or not isinstance(id_to_path, dict)
    ):
        return []

    # Map each node to its parent among the start_ids / expanded set
    parent_in_tree: dict[int, int | None] = {nid: None for nid in start_ids}
    visible_nodes = list(start_ids)
    # Track how many times each node has been expanded (0 = never)
    expanded_counts: dict[str, int] = {str(nid): 0 for nid in start_ids}
    # Track relative depth from start_ids (all start at 0)
    depth_map: dict[str, int] = {str(nid): 0 for nid in start_ids}

    # Candidates: nodes with children that are not yet fully expanded
    candidates = set()
    for node_id in start_ids:
        key = str(node_id)
        if children_map.get(key):
            candidates.add(key)

    while len(visible_nodes) < limit and candidates:
        def sort_key(node_id_str: str) -> tuple[int, int, int]:
            count = expanded_counts.get(node_id_str, 0)
            depth = depth_map.get(node_id_str, 0)
            descendants = subtree_size_map.get(node_id_str, 1)
            return (count, depth, -descendants)

        best_node_str = min(candidates, key=sort_key)

        children = children_map.get(best_node_str, [])
        count = expanded_counts.get(best_node_str, 0)
        offset = count * 3
        to_add = children[offset : offset + 3]
        expanded_counts[best_node_str] = count + 1

        # Remove from candidates if fully expanded
        if offset + 3 >= len(children):
            candidates.discard(best_node_str)

        parent_id = int(best_node_str)
        child_depth = depth_map.get(best_node_str, 0) + 1
        for ans_raw in to_add:
            try:
                ans = int(ans_raw)
            except Exception:
                continue
            if ans not in parent_in_tree:
                visible_nodes.append(ans)
                parent_in_tree[ans] = parent_id
                ans_str = str(ans)
                expanded_counts[ans_str] = 0
                depth_map[ans_str] = child_depth
                if children_map.get(ans_str):
                    candidates.add(ans_str)

    # Count how many children of each visible node are also visible
    listed_children_count: dict[int, int] = {nid: 0 for nid in visible_nodes}
    for nid in visible_nodes:
        pid = parent_in_tree.get(nid)
        if pid is not None and pid in listed_children_count:
            listed_children_count[pid] += 1

    output: list[dict[str, Any]] = []
    for node_id in visible_nodes:
        node_str = str(node_id)
        child_path = id_to_path.get(node_str, f"<missing:{node_id}>")
        child_path = str(child_path)
        total_children = len(children_map.get(node_str, []))
        listed = listed_children_count.get(node_id, 0)

        output.append({
            "node_id": node_id,
            "path": _normalize_path(child_path),
            "parent_id": parent_in_tree.get(node_id),
            "depth": depth_map.get(node_str, 0),
            "listed_children": listed,
            "total_children": total_children,
        })

    return output


def _render_hsn_tree(nodes: list[dict[str, Any]]) -> str:
    """Render expanded HSN nodes as an indented tree.

    Each line: ``<indent><path> (<listed>/<total> children)``
    where indentation reflects parent-child depth.
    """
    lines: list[str] = []
    for node in nodes:
        indent = "  " * node["depth"]
        ratio = f"({node['listed_children']}/{node['total_children']} children)"
        lines.append(f"{indent}{node['path']} {ratio}")
    return "\n".join(lines)


def _build_initial_message(index_data: dict[str, Any]) -> str:
    roots = index_data.get("roots", [])

    if not isinstance(roots, list):
        roots = []

    lines = [
        "Hierarchical Semantic Navigation (HSN) assists with filesystem navigation.",
        "Files are organized in a semantic hierarchy where indentation shows parent-child relationships.",
        "Each branch is a semantic cluster, and a node is semantically closer to its parent than to higher predecessors.",
        "A ratio of listed children to all children under HSN will be provided for each file. A limited number of files will be listed at a time.",
        "Some tools will automatically be augmented with HSN info, which you need for further exploration if the tree is not complete.",
        "As a courtesy, you will be provided an initial HSN tree to help you start navigating.",
        "",
        "HSN tree:",
    ]

    nodes = _expand_hsn_nodes_from_data(index_data, roots)

    if not nodes:
        lines.append("(No document-like files were indexed in this world.)")
    else:
        lines.append(_render_hsn_tree(nodes))

    return "\n".join(lines)


def _populate_index_into_environment(
    index_path: Path, output_dir: Path, env_url: str
) -> None:
    tar_path = output_dir / "hsn.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        data = index_path.read_bytes()
        info = tarfile.TarInfo(name=HSN_INDEX_REL_PATH)
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))

    with open(tar_path, "rb") as f:
        resp = httpx.post(
            f"{env_url}/data/populate",
            files={"archive": ("hsn.tar.gz", f.read(), "application/gzip")},
            params={"subsystem": ".apps_data"},
            timeout=600.0,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to populate HSN data: {resp.text}")


def prepare_hsn_for_world(
    *,
    world_id: str,
    world_zip: Path,
    output_dir: Path,
    env_url: str,
    log: Callable[[str], None],
) -> PreparedHSN:
    cache_dir = _cache_dir_for_world(world_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = cache_dir / "hsn_index.json"
    embeddings_path = cache_dir / "embeddings.npy"
    docs_path = cache_dir / "documents.json"

    if index_path.exists():
        with open(index_path) as f:
            index_data = json.load(f)
        if _index_cache_compatible(index_data):
            log(f"  HSN cache hit: {index_path}")
        else:
            log("  HSN cache mismatch: rebuilding index for current embedding settings")
            index_data, embeddings, docs = _build_hsn_index(
                world_id, world_zip, cache_dir, log
            )
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2, sort_keys=True)
            with open(docs_path, "w") as f:
                json.dump(
                    [
                        {"id": doc["id"], "path": doc["path"], "size": doc["size"]}
                        for doc in docs
                    ],
                    f,
                    indent=2,
                    sort_keys=True,
                )
            np.save(embeddings_path, embeddings)
            log(f"  HSN index saved: {index_path}")
    else:
        log(f"  HSN cache miss: building index for {world_id} with {HSN_EMBEDDING_MODEL}")
        index_data, embeddings, docs = _build_hsn_index(world_id, world_zip, cache_dir, log)
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2, sort_keys=True)
        # Persist source docs and embeddings so we never recompute for same world/model.
        with open(docs_path, "w") as f:
            json.dump(
                [
                    {"id": doc["id"], "path": doc["path"], "size": doc["size"]}
                    for doc in docs
                ],
                f,
                indent=2,
                sort_keys=True,
            )
        np.save(embeddings_path, embeddings)
        log(f"  HSN index saved: {index_path}")

    _populate_index_into_environment(index_path, output_dir, env_url)
    initial_message = _build_initial_message(index_data)
    return PreparedHSN(
        index_path=index_path,
        index_data=index_data,
        initial_message=initial_message,
    )
