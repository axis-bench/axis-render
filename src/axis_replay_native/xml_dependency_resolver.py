"""Recursively resolve file dependencies from MJCF XML."""

from __future__ import annotations

import re
from collections import deque
from typing import Callable

_FILE_ATTR_RE = re.compile(r'file="([^"]+)"')


def extract_referenced_files(xml_text: str) -> list[str]:
    files: set[str] = set()
    for match in _FILE_ATTR_RE.finditer(xml_text):
        path = match.group(1).strip()
        if path:
            files.add(path)
    return list(files)


def normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    path = re.sub(r"^\.?/", "", path)
    parts = path.split("/")
    stack: list[str] = []
    for part in parts:
        if not part or part == ".":
            continue
        if part == "..":
            if stack:
                stack.pop()
            continue
        stack.append(part)
    return "/".join(stack)


def get_path_dir(path: str) -> str:
    idx = path.rfind("/")
    if idx <= 0:
        return ""
    return path[:idx]


def resolve_xml_dependencies(
    entry_xml_path: str,
    xml_loader: Callable[[str], str],
) -> tuple[list[str], list[str]]:
    pending: deque[str] = deque([entry_xml_path])
    seen_xml: set[str] = set()
    xml_paths: list[str] = []
    asset_paths: list[str] = []
    asset_set: set[str] = set()

    while pending:
        xml_path = pending.popleft()
        normalized = normalize_path(xml_path)
        if normalized in seen_xml:
            continue
        seen_xml.add(normalized)
        xml_paths.append(normalized)

        try:
            xml_text = xml_loader(normalized)
        except Exception:
            continue

        referenced = extract_referenced_files(xml_text)
        xml_dir = get_path_dir(normalized)

        for ref in referenced:
            ref_clean = ref.replace("\\", "/").strip()
            if ref_clean.startswith("./"):
                ref_clean = ref_clean[2:]
            if ref_clean.startswith("/"):
                ref_clean = ref_clean[1:]
            if not ref_clean or ref_clean.lower().startswith("http"):
                continue

            if xml_dir:
                combined = normalize_path(f"{xml_dir}/{ref_clean}")
            else:
                combined = normalize_path(ref_clean)

            if combined.endswith(".xml") or combined.endswith(".mjcf"):
                if combined not in seen_xml:
                    pending.append(combined)
            elif combined not in asset_set:
                asset_set.add(combined)
                asset_paths.append(combined)

    return xml_paths, asset_paths
