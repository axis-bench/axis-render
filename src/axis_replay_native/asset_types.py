"""Asset resolution result structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AssetResolution:
    xml_files: dict[str, str] = field(default_factory=dict)
    binary_files: dict[str, bytes] = field(default_factory=dict)
    missing_files: list[str] = field(default_factory=list)
    entry_xml_path: str = ""
