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


def expand_hsn_nodes(start_ids: list[int], limit: int | None = None) -> list[tuple[str, list[int]]]:
    if limit is None:
        limit = int(os.getenv("FS_HSN_EXPAND_LIMIT", "30"))
        
    data = _read_index()
    if data is None:
        return []

    children_map = data.get("children", {})
    subtree_size_map = data.get("subtree_size", {})
    paths_map = data.get("paths", {})
    id_to_path = data.get("id_to_path", {})

    if (
        not isinstance(children_map, dict)
        or not isinstance(subtree_size_map, dict)
        or not isinstance(paths_map, dict)
        or not isinstance(id_to_path, dict)
    ):
        return []

    visible_nodes = list(start_ids)
    expanded_counts = {str(node_id): 0 for node_id in start_ids}

    candidates = set()
    for node_id in start_ids:
        key = str(node_id)
        if children_map.get(key):
            candidates.add(key)

    while len(visible_nodes) < limit and candidates:
        def sort_key(node_id_str):
            node_path = paths_map.get(node_id_str, [])
            depth = len(node_path) - 1 if isinstance(node_path, list) else 0
            descendants = subtree_size_map.get(node_id_str, 1)
            return (depth, -descendants)

        best_node_str = min(candidates, key=sort_key)

        children = children_map.get(best_node_str, [])
        offset = expanded_counts.get(best_node_str, 0)
        to_add = children[offset : offset + 3]
        expanded_counts[best_node_str] = offset + len(to_add)

        for ans_raw in to_add:
            try:
                ans = int(ans_raw)
            except Exception:
                continue
            if ans not in visible_nodes:
                visible_nodes.append(ans)
                ans_str = str(ans)
                if children_map.get(ans_str):
                    candidates.add(ans_str)
                    expanded_counts[ans_str] = 0

        if expanded_counts[best_node_str] >= len(children):
            candidates.remove(best_node_str)

    output: list[tuple[str, list[int]]] = []
    for node_id in visible_nodes:
        node_str = str(node_id)
        child_path = id_to_path.get(node_str, f"<missing:{node_id}>")
        # Ensure it is a string for printing
        child_path = str(child_path)

        raw_path_ids = paths_map.get(node_str, [node_id])
        path_ids: list[int] = []
        if isinstance(raw_path_ids, list):
            for value in raw_path_ids:
                try:
                    path_ids.append(int(value))
                except Exception:
                    continue
        if not path_ids:
            path_ids = [node_id]

        output.append((_normalize_path(child_path), path_ids))
        
    return output
