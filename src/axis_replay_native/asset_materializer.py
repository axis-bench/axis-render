"""Materialize assets to a replay working directory."""

from __future__ import annotations

import logging
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from .asset_store import HttpAssetStore, LocalAssetStore
from .asset_types import AssetResolution
from .xml_dependency_resolver import normalize_path, resolve_xml_dependencies

logger = logging.getLogger(__name__)

_FILE_ATTR_RE = re.compile(r'(file=")([^"]+)(")')
SCOPE_TOKEN = "__axis_scope__"
_PATHLIKE_ATTRS = {"file", "meshdir", "texturedir", "assetdir"}


def _rewrite_xml_file_refs(work_path: Path, xml_relpaths: list[str]) -> None:
    for xml_relpath in xml_relpaths:
        xml_path = work_path / xml_relpath
        if not xml_path.is_file():
            continue

        xml_dir = normalize_path(str(Path(xml_relpath).parent))
        if xml_dir == ".":
            xml_dir = ""

        text = xml_path.read_text(encoding="utf-8")

        def repl(match: re.Match[str]) -> str:
            prefix, raw_path, suffix = match.groups()
            ref_clean = raw_path.replace("\\", "/").strip()
            if not ref_clean or ref_clean.lower().startswith("http") or ref_clean.startswith("/"):
                return match.group(0)

            if xml_dir:
                combined = normalize_path(f"{xml_dir}/{ref_clean}")
            else:
                combined = normalize_path(ref_clean)

            target_path = work_path / combined
            try:
                relpath = os.path.relpath(target_path, start=xml_path.parent).replace("\\", "/")
            except ValueError:
                return match.group(0)
            return f'{prefix}{relpath}{suffix}'

        rewritten = _FILE_ATTR_RE.sub(repl, text)
        if rewritten != text:
            xml_path.write_text(rewritten, encoding="utf-8")


def _sanitize_xml_identifiers(work_path: Path, xml_relpaths: list[str]) -> None:
    for xml_relpath in xml_relpaths:
        xml_path = work_path / xml_relpath
        if not xml_path.is_file():
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        changed = False

        for elem in root.iter():
            for attr_name, attr_value in list(elem.attrib.items()):
                if attr_name in _PATHLIKE_ATTRS:
                    continue
                if "/" not in attr_value:
                    continue

                sanitized_value = attr_value.replace("/", SCOPE_TOKEN)
                if sanitized_value != attr_value:
                    elem.set(attr_name, sanitized_value)
                    changed = True

        if changed:
            tree.write(xml_path, encoding="utf-8")


def repair_materialized_scene(
    work_dir: str | Path,
    entry_filename: str = "scenes/scene.xml",
    *,
    sanitize_identifiers: bool = False,
) -> None:
    work_path = Path(work_dir)
    entry_path = normalize_path(entry_filename)

    def xml_loader(path: str) -> str:
        return (work_path / path).read_text(encoding="utf-8")

    try:
        xml_paths, _ = resolve_xml_dependencies(entry_path, xml_loader)
    except Exception:
        xml_paths = [entry_path] if (work_path / entry_path).is_file() else []

    if xml_paths:
        _rewrite_xml_file_refs(work_path, xml_paths)
        if sanitize_identifiers:
            _sanitize_xml_identifiers(work_path, xml_paths)


def materialize_scene(
    mjcf_xml: str,
    work_dir: str,
    asset_store: HttpAssetStore | LocalAssetStore,
    entry_filename: str = "scenes/scene.xml",
    *,
    sanitize_identifiers: bool = False,
) -> AssetResolution:
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    resolution = AssetResolution(entry_xml_path=entry_filename)
    entry_path = work_path / entry_filename
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text(mjcf_xml, encoding="utf-8")
    resolution.xml_files[entry_filename] = mjcf_xml

    xml_cache: dict[str, str] = {normalize_path(entry_filename): mjcf_xml}

    def xml_loader(path: str) -> str:
        normalized = normalize_path(path)
        if normalized in xml_cache:
            return xml_cache[normalized]

        file_path = work_path / normalized
        if file_path.is_file():
            text = file_path.read_text(encoding="utf-8")
            xml_cache[normalized] = text
            resolution.xml_files[normalized] = text
            return text

        data = asset_store.fetch_file(normalized)
        if data is None:
            raise FileNotFoundError(f"XML not found: {normalized}")

        text = data.decode("utf-8")
        xml_cache[normalized] = text
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")
        resolution.xml_files[normalized] = text
        return text

    xml_paths, asset_paths = resolve_xml_dependencies(entry_filename, xml_loader)
    logger.info(
        "Resolved dependencies: %d XML files, %d asset files",
        len(xml_paths),
        len(asset_paths),
    )

    to_fetch: list[str] = []
    for path in asset_paths:
        normalized = normalize_path(path)
        file_path = work_path / normalized
        if file_path.is_file():
            resolution.binary_files[normalized] = file_path.read_bytes()
        else:
            to_fetch.append(normalized)

    found, missing = asset_store.fetch_files(to_fetch)
    for path, data in found.items():
        file_path = work_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        resolution.binary_files[path] = data

    if missing:
        resolution.missing_files = missing
        logger.warning("Missing assets: %s", missing)

    _rewrite_xml_file_refs(work_path, xml_paths)
    if sanitize_identifiers:
        _sanitize_xml_identifiers(work_path, xml_paths)

    return resolution
