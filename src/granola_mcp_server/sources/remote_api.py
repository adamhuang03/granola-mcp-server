"""Remote API-based document source.

Fetches documents directly from the Granola public API using token authentication.
Handles caching of JSON responses with TTL-based invalidation and retry logic.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib import request
from urllib.error import HTTPError, URLError

from ..errors import GranolaParseError
from ..document_source import DocumentSource


class RemoteApiDocumentSource(DocumentSource):
    """Document source that fetches from the Granola public API.

    Features:
    - Token-based authentication (Bearer grn_...)
    - Cursor-based pagination across all pages up to limit
    - Per-document detail fetching via /v1/notes/{id}?include=transcript
    - Local caching of responses with TTL-based invalidation
    - Retry logic with exponential backoff

    Args:
        token: Bearer token for API authentication (GRANOLA_API_TOKEN).
        api_base: Base URL for the Granola public API.
        cache_dir: Directory for storing cache files.
        cache_ttl_seconds: Time-to-live for cached data (default 24h).
    """

    def __init__(
        self,
        token: str,
        api_base: str = "https://public-api.granola.ai",
        cache_dir: Optional[str | Path] = None,
        cache_ttl_seconds: int = 86400,  # 24 hours
    ):
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.cache_ttl = cache_ttl_seconds

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".granola" / "remote_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> Path:
        """Get cache file path for a given name/key."""
        safe = hashlib.sha256(name.encode()).hexdigest()[:16]
        return self.cache_dir / f"docs_{safe}.json"

    def _is_cache_fresh(self, cache_path: Path) -> bool:
        """Check if cache file is within TTL."""
        if not cache_path.exists():
            return False
        age = time.time() - cache_path.stat().st_mtime
        return age < self.cache_ttl

    def _read_cache(self, cache_path: Path) -> Optional[object]:
        """Read from cache file."""
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _write_cache(self, cache_path: Path, data: object) -> None:
        """Write to cache file."""
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to write cache: {e}")

    def _make_request(self, url: str) -> Dict[str, object]:
        """Execute a GET request against the public API with retry logic.

        Args:
            url: Full URL to fetch.

        Returns:
            Parsed JSON response dict.

        Raises:
            GranolaParseError: For network or parsing errors.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "User-Agent": "Granola-MCP-Server/0.1.0",
        }

        req = request.Request(url, headers=headers, method="GET")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with request.urlopen(req, timeout=30) as response:
                    raw_data = response.read()
                    try:
                        data = json.loads(raw_data.decode("utf-8"))
                        return data
                    except Exception as e:
                        raise GranolaParseError(
                            f"Failed to parse JSON: {e}",
                            {"attempt": attempt + 1, "url": url},
                        ) from e

            except HTTPError as e:
                error_code = e.code
                error_body = e.read().decode("utf-8", errors="replace")

                if error_code == 401:
                    raise GranolaParseError(
                        "Invalid or expired token. Please check GRANOLA_API_TOKEN.",
                        {"status": 401},
                    ) from e
                elif error_code == 403:
                    raise GranolaParseError(
                        "Access forbidden. Check your token permissions.",
                        {"status": 403},
                    ) from e
                elif error_code == 429:
                    if attempt < max_retries - 1:
                        time.sleep((2 ** attempt) * 1.0)
                        continue
                    raise GranolaParseError(
                        "Rate limit exceeded. Please try again later.",
                        {"status": 429, "attempt": attempt + 1},
                    ) from e
                elif 500 <= error_code < 600:
                    if attempt < max_retries - 1:
                        time.sleep((2 ** attempt) * 1.0)
                        continue
                    raise GranolaParseError(
                        f"Server error: {error_code}",
                        {"status": error_code, "body": error_body, "attempt": attempt + 1},
                    ) from e
                else:
                    raise GranolaParseError(
                        f"HTTP error: {error_code}",
                        {"status": error_code, "body": error_body},
                    ) from e

            except URLError as e:
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 1.0)
                    continue
                raise GranolaParseError(
                    f"Network error: {e.reason}",
                    {"attempt": attempt + 1},
                ) from e

            except GranolaParseError:
                raise

            except Exception as e:
                raise GranolaParseError(
                    f"Unexpected error: {e}",
                    {"attempt": attempt + 1},
                ) from e

        raise GranolaParseError("Failed after max retries")

    def _fetch_from_api(
        self,
        limit: int = 100,
        created_after: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Fetch all notes from GET /v1/notes, following cursor pagination.

        Collects pages until there are no more results or `limit` is reached.

        Args:
            limit: Maximum total notes to return across all pages.
            created_after: Optional ISO 8601 lower bound for created_at.

        Returns:
            List of note dicts from the API.
        """
        all_notes: List[Dict[str, object]] = []
        cursor: Optional[str] = None
        page_size = min(30, limit)  # API max page size is 30

        while len(all_notes) < limit:
            params = [f"page_size={page_size}"]
            if created_after:
                params.append(f"created_after={created_after}")
            if cursor:
                params.append(f"cursor={cursor}")

            url = f"{self.api_base}/v1/notes?{'&'.join(params)}"
            data = self._make_request(url)

            notes = data.get("notes", [])
            if not isinstance(notes, list):
                raise GranolaParseError(
                    "Invalid response format: 'notes' field is not a list"
                )

            all_notes.extend(notes)

            has_more = data.get("hasMore", False)
            cursor = data.get("cursor")

            if not has_more or not cursor:
                break

        return all_notes[:limit]

    def get_documents(
        self,
        *,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include_last_viewed_panel: bool = True,
        force: bool = False,
    ) -> List[Dict[str, object]]:
        """Fetch notes from the public API with caching.

        The `offset` and `include_last_viewed_panel` parameters are accepted
        for interface compatibility but are not used by the public API.

        Args:
            limit: Maximum notes to fetch (default 100).
            offset: Ignored (public API uses cursor pagination).
            include_last_viewed_panel: Ignored (not supported by public API).
            force: Bypass cache and fetch fresh data.

        Returns:
            List of note dictionaries.
        """
        limit = limit or 100

        cache_key = f"notes_limit{limit}"
        cache_path = self._get_cache_path(cache_key)

        if not force and self._is_cache_fresh(cache_path):
            cached = self._read_cache(cache_path)
            if isinstance(cached, list):
                return cached

        notes = self._fetch_from_api(limit=limit)

        self._write_cache(cache_path, notes)
        return notes

    def get_document_by_id(
        self, doc_id: str, *, force: bool = False
    ) -> Optional[Dict[str, object]]:
        """Fetch a single note by ID from GET /v1/notes/{id}?include=transcript.

        Includes full detail: attendees, summary_text, summary_markdown,
        transcript, calendar_event, folder_membership.

        Args:
            doc_id: Note ID (not_xxx format).
            force: Bypass cache and fetch fresh data.

        Returns:
            Note dictionary with full detail, or None on 404.
        """
        cache_path = self._get_cache_path(f"note_{doc_id}")

        if not force and self._is_cache_fresh(cache_path):
            cached = self._read_cache(cache_path)
            if isinstance(cached, dict):
                return cached

        url = f"{self.api_base}/v1/notes/{doc_id}?include=transcript"
        try:
            data = self._make_request(url)
        except GranolaParseError as e:
            context = e.args[1] if len(e.args) > 1 else {}
            if isinstance(context, dict) and context.get("status") == 404:
                return None
            raise

        if not isinstance(data, dict):
            return None

        self._write_cache(cache_path, data)
        return data

    def refresh_cache(self) -> None:
        """Clear all cache files to force refresh on next request."""
        for cache_file in self.cache_dir.glob("docs_*.json"):
            try:
                cache_file.unlink()
            except Exception:
                pass

    def get_cache_info(self) -> Dict[str, object]:
        """Get information about the remote cache state."""
        cache_files = list(self.cache_dir.glob("docs_*.json"))
        total_size = sum(f.stat().st_size for f in cache_files)

        oldest_cache = None
        if cache_files:
            oldest = min(cache_files, key=lambda f: f.stat().st_mtime)
            oldest_cache = datetime.fromtimestamp(
                oldest.stat().st_mtime, tz=timezone.utc
            ).isoformat()

        return {
            "source": "remote_api",
            "api_base": self.api_base,
            "cache_dir": str(self.cache_dir),
            "cache_files_count": len(cache_files),
            "total_cache_size_bytes": total_size,
            "cache_ttl_seconds": self.cache_ttl,
            "oldest_cache_ts": oldest_cache,
        }
