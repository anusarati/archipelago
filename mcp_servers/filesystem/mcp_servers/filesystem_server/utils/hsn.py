import json
import os
import threading
from typing import Any

FS_HSN_ENABLED = os.getenv("FS_HSN_ENABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
FS_HSN_INDEX_PATH = os.getenv("FS_HSN_INDEX_PATH", "/.apps_data/hsn/current_world.json")

_LOCK = threading.Lock()
_CACHE_DATA: dict[str, Any] | None = None
_CACHE_MTIME_NS: int | None = None


def _normalize_path(path: str) -> str:
    normalized = os.path.normpath("/" + path.lstrip("/"))
    return normalized if normalized != "." else "/"


def _read_index() -> dict[str, Any] | None:
    global _CACHE_DATA, _CACHE_MTIME_NS

    if not FS_HSN_ENABLED:
        return None

    try:
        stat = os.stat(FS_HSN_INDEX_PATH)
    except OSError:
        return None

    with _LOCK:
        if _CACHE_DATA is not None and _CACHE_MTIME_NS == stat.st_mtime_ns:
            return _CACHE_DATA

        try:
            with open(FS_HSN_INDEX_PATH) as f:
                raw = json.load(f)
        except Exception:
            return None

        if not isinstance(raw, dict):
            return None

        path_to_id = raw.get("path_to_id", {})
        if not isinstance(path_to_id, dict):
            path_to_id = {}
        normalized_path_to_id: dict[str, int] = {}
        for path, node_id in path_to_id.items():
            try:
                normalized_path_to_id[_normalize_path(str(path))] = int(node_id)
            except Exception:
                continue
        raw["path_to_id"] = normalized_path_to_id

        _CACHE_DATA = raw
        _CACHE_MTIME_NS = stat.st_mtime_ns
        return _CACHE_DATA


def hsn_mode_enabled() -> bool:
    return _read_index() is not None


def hsn_path_ids(path: str) -> list[int] | None:
    data = _read_index()
    if data is None:
        return None

    path_to_id = data.get("path_to_id", {})
    if not isinstance(path_to_id, dict):
        return None

    file_id = path_to_id.get(_normalize_path(path))
    if file_id is None:
        return None

    paths = data.get("paths", {})
    if not isinstance(paths, dict):
        return [int(file_id)]

    raw_ids = paths.get(str(file_id))
    if not isinstance(raw_ids, list):
        return [int(file_id)]

    result: list[int] = []
    for value in raw_ids:
        try:
            result.append(int(value))
        except Exception:
            continue
    return result or [int(file_id)]


def annotate_path(path: str) -> tuple[str, list[int]]:
    ids = hsn_path_ids(path)
    if not ids:
        return "[HSN: unavailable]", []
    return f"[HSN: {ids}]", ids


def hsn_children(path: str, limit: int = 10) -> list[tuple[str, list[int]]]:
    data = _read_index()
    if data is None:
        return []

    path_to_id = data.get("path_to_id", {})
    children = data.get("children", {})
    id_to_path = data.get("id_to_path", {})
    paths = data.get("paths", {})
    if (
        not isinstance(path_to_id, dict)
        or not isinstance(children, dict)
        or not isinstance(id_to_path, dict)
        or not isinstance(paths, dict)
    ):
        return []

    file_id = path_to_id.get(_normalize_path(path))
    if file_id is None:
        return []

    child_ids = children.get(str(file_id), [])
    if not isinstance(child_ids, list):
        return []

    output: list[tuple[str, list[int]]] = []
    for child_id_raw in child_ids[: max(0, limit)]:
        try:
            child_id = int(child_id_raw)
        except Exception:
            continue

        child_path = id_to_path.get(str(child_id))
        if not isinstance(child_path, str):
            continue

        raw_path_ids = paths.get(str(child_id), [child_id])
        path_ids: list[int] = []
        if isinstance(raw_path_ids, list):
            for value in raw_path_ids:
                try:
                    path_ids.append(int(value))
                except Exception:
                    continue
        if not path_ids:
            path_ids = [child_id]

        output.append((_normalize_path(child_path), path_ids))
    return output


def render_id_map(ids: set[int]) -> str:
    if not ids:
        return ""
    data = _read_index()
    if data is None:
        return ""
    id_to_path = data.get("id_to_path", {})
    if not isinstance(id_to_path, dict):
        return ""

    mapping = {
        str(node_id): str(id_to_path.get(str(node_id), f"<missing:{node_id}>"))
        for node_id in sorted(ids)
    }
    return json.dumps(mapping, indent=2, sort_keys=True)
