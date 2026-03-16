import mimetypes
import os
from typing import Annotated

from pydantic import Field
from utils.decorators import make_async_background
from utils.hsn import (
    FS_HSN_ENABLED,
    annotate_path,
    hsn_children,
    hsn_mode_enabled,
    render_id_map,
)

FS_ROOT = os.getenv("APP_FS_ROOT", "/filesystem")
_LIST_FILES_DESCRIPTION = (
    "Absolute path within the sandbox filesystem to list. Must start with '/'. This is "
    "NOT a system path - '/' refers to the sandbox root. Default: '/' (sandbox root). "
    "Example: '/documents' or '/data/uploads'. Returns a newline-separated string where "
    "each line describes one entry: \"'name' (folder)\\n\" for directories, "
    "\"'name' (mime/type file) N bytes\\n\" for files. MIME type is guessed from "
    "extension ('unknown' if undetectable). Returns '[not found: path]', "
    "'[permission denied: path]', or '[not a directory: path]' for errors. Returns "
    "'No items found' for empty directories."
)
if FS_HSN_ENABLED:
    _LIST_FILES_DESCRIPTION += (
        " In HSN mode, passing a file path returns the file's top-10 HSN children (if "
        "any) plus an HSN id map."
    )


def _resolve_under_root(p: str | None) -> str:
    """Map any incoming path to the sandbox root."""
    if not p or p == "/":
        return FS_ROOT
    rel = os.path.normpath(p).lstrip(os.sep)
    return os.path.join(FS_ROOT, rel)


def _to_relative_path(absolute_path: str) -> str:
    real_root = os.path.realpath(FS_ROOT)
    real_path = os.path.realpath(absolute_path)
    if real_path == real_root:
        return "/"
    if real_path.startswith(real_root + os.sep):
        rel = real_path[len(real_root) :]
        return rel if rel.startswith("/") else "/" + rel
    return absolute_path


@make_async_background
def list_files(
    path: Annotated[
        str,
        Field(
            description=_LIST_FILES_DESCRIPTION
        ),
    ] = "/",
) -> str:
    """List files and folders in a path; each entry shows name and type (file/folder). Use to browse a directory."""
    base = _resolve_under_root(path)
    hsn_enabled = hsn_mode_enabled()

    if not os.path.exists(base):
        return f"[not found: {path}]\n"
    if not os.path.isdir(base):
        if hsn_enabled and os.path.isfile(base):
            file_rel = _to_relative_path(base)
            file_annotation, file_ids = annotate_path(file_rel)
            ids_used = set(file_ids)
            children = hsn_children(file_rel, limit=10)
            if not children:
                lines = [
                    f"'{file_rel}' is a file {file_annotation}",
                    "No HSN children found for this file.",
                ]
            else:
                lines = [
                    f"'{file_rel}' is a file {file_annotation}",
                    f"Top {len(children)} HSN children:",
                ]
                for child_path, child_ids in children:
                    ids_used.update(child_ids)
                    lines.append(f"- {child_path} [HSN: {child_ids}]")

            id_map = render_id_map(ids_used)
            if id_map:
                lines.extend(["", "HSN id map:", id_map])
            return "\n".join(lines)
        return f"[not a directory: {path}]\n"

    items = ""
    ids_used: set[int] = set()
    try:
        with os.scandir(base) as entries:
            for entry in entries:
                if entry.is_dir():
                    items += f"'{entry.name}' (folder)\n"
                elif entry.is_file():
                    mimetype, _ = mimetypes.guess_type(entry.path)
                    stat_result = entry.stat()
                    if hsn_enabled:
                        entry_rel = _to_relative_path(entry.path)
                        annotation, ids = annotate_path(entry_rel)
                        ids_used.update(ids)
                        items += (
                            f"'{entry.name}' ({mimetype or 'unknown'} file) "
                            f"{stat_result.st_size} bytes {annotation}\n"
                        )
                    else:
                        items += (
                            f"'{entry.name}' ({mimetype or 'unknown'} file) "
                            f"{stat_result.st_size} bytes\n"
                        )
    except FileNotFoundError:
        items = f"[not found: {path}]\n"
    except PermissionError:
        items = f"[permission denied: {path}]\n"
    except NotADirectoryError:
        items = f"[not a directory: {path}]\n"

    if not items:
        if not path or path == "/":
            items = (
                "Directory is empty. If you expected files here, use a more specific "
                "path (e.g., '/documents', '/data'). The root '/' maps to the sandbox "
                "root which may not contain files at the top level."
            )
        else:
            items = f"No items found in '{path}'"

    if items and hsn_enabled and ids_used:
        id_map = render_id_map(ids_used)
        if id_map:
            items += f"\nHSN id map:\n{id_map}\n"

    return items
