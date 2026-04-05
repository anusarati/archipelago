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


def _expand_hsn_nodes_impl(
    children_map: dict,
    subtree_size_map: dict,
    paths_map: dict,
    id_to_path: dict,
    start_ids: list[int],
    limit: int,
) -> list[dict]:
    """Core expansion logic shared by expand_hsn_nodes and hsn_pipeline.

    Returns a list of dicts:
        {
            "node_id": int,
            "path": str,           # filesystem path
            "parent_id": int|None, # HSN parent node id (None for roots/start_ids)
            "depth": int,          # depth in the visible tree (0 for start_ids)
            "listed_children": int, # number of children listed in output
            "total_children": int,  # total number of children
        }
    """
    visible_nodes = set()
    parent_in_tree: dict[int, int | None] = {}
    depth_map: dict[str, int] = {}
    distance_map: dict[str, int] = {}
    up_distance_map: dict[str, int] = {}
    is_prio: dict[str, bool] = {}

    ancestors_only = set()
    initial_nodes = set(start_ids)
    
    # First, collect all paths for start_ids to add ancestors unconditionally
    for nid in start_ids:
        path_raw = paths_map.get(str(nid), [nid])
        path_ids = []
        for x in path_raw:
            try:
                path_ids.append(int(x))
            except Exception:
                pass
        if not path_ids:
            path_ids = [nid]
            
        for val in path_ids:
            visible_nodes.add(val)
            if val not in initial_nodes:
                ancestors_only.add(val)
                
        # Link absolute paths and store distances
        path_len = len(path_ids)
        for i, val in enumerate(path_ids):
            depth_map[str(val)] = i
            dist = path_len - 1 - i
            val_str = str(val)
            if val_str not in distance_map or dist < distance_map[val_str]:
                distance_map[val_str] = dist
            if val_str not in up_distance_map or dist < up_distance_map[val_str]:
                up_distance_map[val_str] = dist
                
            if i > 0:
                parent_in_tree[val] = path_ids[i-1]
            else:
                if val not in parent_in_tree:
                    parent_in_tree[val] = None

    # Priority applies to initial nodes and their descendants. Ancestors are not prio.
    for val in visible_nodes:
        val_str = str(val)
        if val in initial_nodes:
            is_prio[val_str] = True
        else:
            is_prio[val_str] = False

    # Candidates: nodes with children that we can expand
    candidates = set()
    expanded_counts: dict[str, int] = {}
    for val in visible_nodes:
        val_str = str(val)
        expanded_counts[val_str] = 0
        if children_map.get(val_str):
            candidates.add(val_str)
            
    countable_count = sum(1 for v in visible_nodes if v not in ancestors_only)

    while countable_count < limit and candidates:
        def sort_key(node_id_str: str) -> tuple[int, int, int, int, int]:
            is_not_prio = 1 if not is_prio.get(node_id_str, False) else 0
            count = expanded_counts.get(node_id_str, 0)
            dist = distance_map.get(node_id_str, 0)
            up_dist = up_distance_map.get(node_id_str, 0)
            descendants = subtree_size_map.get(node_id_str, 1)
            return (is_not_prio, count, dist, up_dist, -descendants)

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
        parent_prio = is_prio.get(best_node_str, False)
        child_depth = depth_map.get(best_node_str, 0) + 1
        child_dist = distance_map.get(best_node_str, 0) + 1
        child_up_dist = up_distance_map.get(best_node_str, 0)
        
        for ans_raw in to_add:
            try:
                ans = int(ans_raw)
            except Exception:
                continue
            if ans not in parent_in_tree:
                visible_nodes.add(ans)
                parent_in_tree[ans] = parent_id
                ans_str = str(ans)
                expanded_counts[ans_str] = 0
                depth_map[ans_str] = child_depth
                distance_map[ans_str] = child_dist
                up_distance_map[ans_str] = child_up_dist
                is_prio[ans_str] = parent_prio
                if children_map.get(ans_str):
                    candidates.add(ans_str)
                
                if ans not in ancestors_only:
                    countable_count += 1

    visible_nodes = list(visible_nodes)
    listed_children_count: dict[int, int] = {nid: 0 for nid in visible_nodes}
    for nid in visible_nodes:
        pid = parent_in_tree.get(nid)
        if pid is not None and pid in listed_children_count:
            listed_children_count[pid] += 1

    # Build output dicts keyed by node_id
    node_dicts: dict[int, dict] = {}
    for node_id in visible_nodes:
        node_str = str(node_id)
        child_path = id_to_path.get(node_str, f"<missing:{node_id}>")
        child_path = str(child_path)
        total_children = len(children_map.get(node_str, []))
        listed = listed_children_count.get(node_id, 0)

        node_dicts[node_id] = {
            "node_id": node_id,
            "path": _normalize_path(child_path),
            "parent_id": parent_in_tree.get(node_id),
            "depth": depth_map.get(node_str, 0),
            "listed_children": listed,
            "total_children": total_children,
        }

    # Build children lookup for DFS ordering
    visible_children: dict[int, list[int]] = {}
    for nid in visible_nodes:
        pid = parent_in_tree.get(nid)
        if pid is not None:
            visible_children.setdefault(pid, []).append(nid)

    for pid in visible_children:
        visible_children[pid].sort(key=lambda x: str(id_to_path.get(str(x), str(x))))

    # DFS to produce tree-ordered output
    output: list[dict] = []
    roots = [nid for nid in visible_nodes if parent_in_tree.get(nid) is None]
    roots.sort(key=lambda x: str(id_to_path.get(str(x), str(x))))

    def dfs(nid: int) -> None:
        output.append(node_dicts[nid])
        for child in visible_children.get(nid, []):
            dfs(child)

    for root in roots:
        dfs(root)

    return output


def render_hsn_tree(nodes: list[dict]) -> str:
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


def expand_hsn_nodes(start_ids: list[int], limit: int | None = None) -> list[dict]:
    """Expand HSN nodes from the on-disk index.

    Returns a list of dicts with keys:
        node_id, path, parent_id, depth, listed_children, total_children
    """
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

    return _expand_hsn_nodes_impl(
        children_map, subtree_size_map, paths_map, id_to_path,
        start_ids, limit,
    )
