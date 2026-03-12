from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx
import litellm
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
HSN_EMBEDDING_API_KEY = os.environ.get("HSN_EMBEDDING_API_KEY")
HSN_EMBEDDING_BASE_URL = os.environ.get("HSN_EMBEDDING_BASE_URL")
HSN_MAX_TEXT_CHARS = int(os.environ.get("HSN_MAX_TEXT_CHARS", "12000"))
HSN_MAX_FILES = int(os.environ.get("HSN_MAX_FILES", "5000"))

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


def _build_converter() -> Any:
    try:
        from markitdown import MarkItDown  # import locally for optional dependency
    except Exception as exc:
        raise RuntimeError(
            "HSN mode requires markitdown. Install dependencies with `cd agents && uv sync`."
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


def _extract_text(path: str, data: bytes, converter: Any) -> str:
    try:
        document = converter.convert(io.BytesIO(data))
        text = getattr(document, "text_content", None)
        if not text:
            text = str(document) if document is not None else ""
        if text:
            return text[:HSN_MAX_TEXT_CHARS]
    except Exception:
        pass

    ext = _extension(path)
    if ext in TEXT_EXTENSIONS:
        return data.decode("utf-8", errors="ignore")[:HSN_MAX_TEXT_CHARS]
    return data.decode("latin-1", errors="ignore")[:HSN_MAX_TEXT_CHARS]


def _collect_documents(world_zip: Path, log: Callable[[str], None]) -> list[dict[str, Any]]:
    converter = _build_converter()
    docs: list[dict[str, Any]] = []
    with zipfile.ZipFile(world_zip, "r") as zf:
        names = sorted(
            n for n in zf.namelist() if n.startswith("filesystem/") and not n.endswith("/")
        )
        for name in names:
            rel_path = _normalize_path(name[len("filesystem/") :])
            if not _document_like(rel_path):
                continue
            data = zf.read(name)
            text = _extract_text(rel_path, data, converter)
            docs.append(
                {
                    "id": str(len(docs) + 1),
                    "path": rel_path,
                    "size": len(data),
                    "text": text,
                }
            )
            if len(docs) >= HSN_MAX_FILES:
                break

    log(f"  HSN: collected {len(docs)} document-like files")
    return docs


def _extract_embeddings(texts: list[str], log: Callable[[str], None]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    extra_args = _load_embedding_extra_args()
    vectors: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, HSN_EMBEDDING_BATCH_SIZE):
        end = min(total, start + HSN_EMBEDDING_BATCH_SIZE)
        batch = texts[start:end]
        log(f"  HSN: embedding batch {start + 1}-{end}/{total}")
        kwargs: dict[str, Any] = {
            "model": HSN_EMBEDDING_MODEL,
            "input": batch,
            **extra_args,
        }
        if HSN_EMBEDDING_API_KEY:
            kwargs["api_key"] = HSN_EMBEDDING_API_KEY
        if HSN_EMBEDDING_BASE_URL:
            kwargs["api_base"] = HSN_EMBEDDING_BASE_URL

        response = litellm.embedding(**kwargs)

        data = getattr(response, "data", None)
        if data is None and isinstance(response, dict):
            data = response.get("data")
        if not isinstance(data, list) or len(data) != len(batch):
            raise RuntimeError("Unexpected embedding response shape from LiteLLM")

        for item in data:
            if isinstance(item, dict):
                embedding = item.get("embedding")
            else:
                embedding = getattr(item, "embedding", None)
            if not isinstance(embedding, list):
                raise RuntimeError("Missing embedding vector in LiteLLM response")
            vectors.append([float(v) for v in embedding])

    return np.array(vectors, dtype=np.float32)


def _sanitize_model_name(model: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    return slug.strip("_") or "default_model"


def _cache_dir_for_world(world_id: str) -> Path:
    return HSN_CACHE_ROOT / world_id / _sanitize_model_name(HSN_EMBEDDING_MODEL)


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
    world_id: str, world_zip: Path, log: Callable[[str], None]
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    docs = _collect_documents(world_zip, log)
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

    embed_inputs = [f"path: {doc['path']}\n{doc['text']}" for doc in docs]
    embeddings = _extract_embeddings(embed_inputs, log)
    builder = _build_hsn_builder()
    graph = builder.build(embeddings, [doc["id"] for doc in docs])
    index = _graph_to_index(graph, docs)
    index["world_id"] = world_id
    return index, embeddings, docs


def _build_initial_message(index_data: dict[str, Any]) -> str:
    roots = index_data.get("roots", [])
    id_to_path = index_data.get("id_to_path", {})
    paths = index_data.get("paths", {})
    subtree = index_data.get("subtree_size", {})

    if not isinstance(roots, list):
        roots = []
    if not isinstance(id_to_path, dict):
        id_to_path = {}
    if not isinstance(paths, dict):
        paths = {}
    if not isinstance(subtree, dict):
        subtree = {}

    lines = [
        "Hierarchical Semantic Navigation (HSN) is enabled for filesystem exploration.",
        "Each file can be referenced by an HSN path represented as a list of integer file IDs from a top-level root file to a descendant file.",
        "Along a path, files are ordered from most central to least central, each branch is a semantic cluster, and a node is semantically closer to its parent than to higher predecessors.",
        "",
        "Top-level files by subtree size (top 10):",
    ]

    used_ids: set[int] = set()
    top_roots = roots[:10]
    for i, node_id in enumerate(top_roots, start=1):
        key = str(node_id)
        path_ids = paths.get(key, [node_id])
        if not isinstance(path_ids, list):
            path_ids = [node_id]
        cleaned_ids = [int(v) for v in path_ids]
        used_ids.update(cleaned_ids)
        lines.append(
            f"{i}. {id_to_path.get(key, f'<missing:{node_id}>')} | HSN path: {cleaned_ids} | subtree_size={subtree.get(key, 1)}"
        )

    if not top_roots:
        lines.append("(No document-like files were indexed in this world.)")

    id_map = {
        str(node_id): str(id_to_path.get(str(node_id), f"<missing:{node_id}>"))
        for node_id in sorted(used_ids)
    }
    lines.extend(["", "HSN ID dictionary for IDs shown above:", json.dumps(id_map, indent=2, sort_keys=True)])
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
        log(f"  HSN cache hit: {index_path}")
    else:
        log(f"  HSN cache miss: building index for {world_id} with {HSN_EMBEDDING_MODEL}")
        index_data, embeddings, docs = _build_hsn_index(world_id, world_zip, log)
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
