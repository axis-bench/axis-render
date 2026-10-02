"""Asset store supporting local directory or direct HTTP fetching."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import httpx

from .xml_dependency_resolver import normalize_path

logger = logging.getLogger(__name__)

DEFAULT_ASSET_BASE_URL: str | None = None  # no default: pass an explicit base URL
MAX_RETRIES = 3
RETRY_DELAY = 2.0


def _ensure_trailing_slash(url: str) -> str:
    return url if url.endswith("/") else url + "/"


class LocalAssetStore:
    def __init__(self, asset_dir: str):
        self.asset_dir = Path(asset_dir)
        if not self.asset_dir.is_dir():
            raise FileNotFoundError(f"Asset directory not found: {asset_dir}")
        logger.info("Using local asset store: %s", self.asset_dir)

    def _build_candidates(self, file_path: str) -> list[str]:
        normalized = normalize_path(file_path)
        candidates = [normalized]
        prefix = "mujoco-assets/"
        if normalized.startswith(prefix):
            candidates.append(normalized[len(prefix):])
        if not normalized.startswith("scenes/"):
            candidates.append("scenes/" + normalized)
        return candidates

    def fetch_file(self, file_path: str) -> bytes | None:
        normalized = normalize_path(file_path)
        for candidate in self._build_candidates(normalized):
            full_path = self.asset_dir / candidate
            if full_path.is_file():
                return full_path.read_bytes()
        return None

    def fetch_files(self, file_paths: list[str]) -> tuple[dict[str, bytes], list[str]]:
        found: dict[str, bytes] = {}
        missing: list[str] = []
        for fp in file_paths:
            normalized = normalize_path(fp)
            data = self.fetch_file(normalized)
            if data is not None:
                found[normalized] = data
            else:
                missing.append(normalized)
        return found, missing


class HttpAssetStore:
    def __init__(self, asset_base_url: str, timeout: float = 300.0):
        if not asset_base_url:
            raise ValueError("HttpAssetStore needs an explicit asset_base_url")
        self.asset_base_url = _ensure_trailing_slash(asset_base_url)
        cache_dir = os.getenv("AXIS_REPLAY_ASSET_CACHE_DIR") or os.getenv("AXIS_ASSET_CACHE_DIR")
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else Path.cwd() / "tmp" / "axis_replay_asset_cache"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Could not create asset cache directory: %s", self.cache_dir, exc_info=True)
            self.cache_dir = None
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=timeout, write=30.0, pool=30.0),
            follow_redirects=True,
            trust_env=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpAssetStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _fetch_with_retry(self, url: str) -> httpx.Response | None:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._client.get(url)
                if resp.status_code in (200, 403):
                    return resp
                logger.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
                logger.warning(
                    "Fetch failed for %s (attempt %d/%d): %s",
                    url,
                    attempt,
                    MAX_RETRIES,
                    exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
        return None

    def fetch_file(self, file_path: str) -> bytes | None:
        normalized = normalize_path(file_path)
        cache_path = self.cache_dir / normalized if self.cache_dir is not None else None
        if cache_path is not None and cache_path.is_file():
            try:
                return cache_path.read_bytes()
            except OSError:
                logger.warning("Could not read cached asset: %s", cache_path, exc_info=True)
        url = self.asset_base_url + normalized
        logger.info("Fetching: %s", url)
        resp = self._fetch_with_retry(url)
        if resp is not None and resp.status_code == 200:
            if cache_path is not None:
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
                    tmp_path.write_bytes(resp.content)
                    tmp_path.replace(cache_path)
                except OSError:
                    logger.warning("Could not write cached asset: %s", cache_path, exc_info=True)
            return resp.content
        return None

    def fetch_files(self, file_paths: list[str]) -> tuple[dict[str, bytes], list[str]]:
        found: dict[str, bytes] = {}
        missing: list[str] = []
        for fp in file_paths:
            normalized = normalize_path(fp)
            data = self.fetch_file(normalized)
            if data is not None:
                found[normalized] = data
            else:
                missing.append(normalized)
        logger.info("Fetched %d/%d files", len(found), len(found) + len(missing))
        return found, missing
