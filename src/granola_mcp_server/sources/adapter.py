"""Adapter to bridge DocumentSource with GranolaParser interface.

This adapter allows the existing tools and code that expect GranolaParser
to work seamlessly with any DocumentSource implementation.

Public API note shapes (from GET /v1/notes and GET /v1/notes/{id}?include=transcript):
  List shape:  { id, title, owner {name, email}, created_at, updated_at }
  Detail shape: adds attendees [{name, email}], summary_text, summary_markdown,
                transcript [{speaker, text, start_time, end_time}],
                calendar_event, folder_membership
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..document_source import DocumentSource
from ..parser import MeetingDict


class DocumentSourceAdapter:
    """Adapter that presents a DocumentSource as a parser-like interface.

    This allows existing code that expects GranolaParser methods to work
    with any DocumentSource implementation (local or remote).

    Args:
        source: The underlying document source.
    """

    def __init__(self, source: DocumentSource):
        self._source = source
        self._cache: Optional[Dict[str, Any]] = None
        self._loaded_at: Optional[datetime] = None

    def load_cache(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load documents into a cache structure.

        Mimics the GranolaParser.load_cache() behavior.
        """
        if self._cache is not None and not force_reload:
            return self._cache

        docs = self._source.get_documents(force=force_reload)

        # Convert list to dict keyed by id
        documents_dict: Dict[str, Dict[str, Any]] = {}
        for doc in docs:
            if isinstance(doc, dict):
                doc_id = doc.get("id")
                if doc_id:
                    documents_dict[str(doc_id)] = doc

        self._cache = {
            "state": {
                "documents": documents_dict,
            }
        }
        self._loaded_at = datetime.now(timezone.utc)

        return self._cache

    def reload(self) -> Dict[str, Any]:
        """Force reload from source."""
        return self.load_cache(force_reload=True)

    def get_meetings(self, debug: bool = False) -> List[MeetingDict]:
        """Get meetings in the normalized MeetingDict format.

        Maps public API note fields to the MeetingDict schema:
          - attendees[].name  -> participants
          - summary_markdown  -> notes (preferred), summary_text as fallback
          - created_at        -> start_ts
          - folder_membership -> folder_id / folder_name
        """
        cache = self.load_cache()
        state = cache.get("state", {})
        documents = state.get("documents", {})

        meetings: List[MeetingDict] = []

        if not isinstance(documents, dict):
            return meetings

        for doc_key, doc in documents.items():
            if not isinstance(doc, dict):
                continue

            meeting_id = str(doc.get("id") or doc_key)
            title = doc.get("title") or "Untitled Meeting"

            # Timestamp from created_at (ISO 8601 string from public API)
            start_ts = doc.get("created_at") or ""
            if not isinstance(start_ts, str):
                start_ts = str(start_ts) if start_ts else ""

            # Participants from attendees list [{name, email}]
            # Also fall back to owner field present on list endpoint
            participants: List[str] = []
            seen: set = set()

            attendees = doc.get("attendees", [])
            if isinstance(attendees, list):
                for att in attendees:
                    if isinstance(att, dict):
                        name = att.get("name") or att.get("email")
                        if name and name not in seen:
                            participants.append(str(name))
                            seen.add(name)

            # If no attendees yet, try owner (available on list endpoint)
            if not participants:
                owner = doc.get("owner")
                if isinstance(owner, dict):
                    name = owner.get("name") or owner.get("email")
                    if name and name not in seen:
                        participants.append(str(name))
                        seen.add(name)

            # Notes: prefer summary_markdown, fall back to summary_text
            notes: Optional[str] = None
            summary_markdown = doc.get("summary_markdown")
            summary_text = doc.get("summary_text")
            if isinstance(summary_markdown, str) and summary_markdown.strip():
                notes = summary_markdown
            elif isinstance(summary_text, str) and summary_text.strip():
                notes = summary_text

            # Folder from folder_membership {id, name} if present
            folder_id: Optional[str] = None
            folder_name: Optional[str] = None
            folder_membership = doc.get("folder_membership")
            if isinstance(folder_membership, dict):
                folder_id = folder_membership.get("id") or folder_membership.get("folder_id")
                folder_name = folder_membership.get("name") or folder_membership.get("title")
                if folder_id is not None:
                    folder_id = str(folder_id)
                if folder_name is not None:
                    folder_name = str(folder_name)

            meeting: MeetingDict = {
                "id": meeting_id,
                "title": title,
                "start_ts": start_ts,
                "end_ts": None,
                "participants": participants,
                "platform": None,
                "notes": notes,
                "overview": None,
                "summary": None,
                "folder_id": folder_id,
                "folder_name": folder_name,
            }

            meetings.append(meeting)

        # Sort by start_ts descending
        meetings.sort(key=lambda x: x.get("start_ts") or "", reverse=True)

        return meetings

    def get_meeting_by_id(self, meeting_id: str) -> Optional[MeetingDict]:
        """Get a single meeting by ID, fetching full detail from the API."""
        # Try to get the full detail document (includes attendees, summary_markdown, transcript)
        try:
            detail = self._source.get_document_by_id(meeting_id)
        except Exception:
            detail = None

        if detail is not None:
            # Build a temporary adapter state with just this document to normalize it
            tmp_docs = {meeting_id: detail}
            tmp_cache = {"state": {"documents": tmp_docs}}
            old_cache = self._cache
            self._cache = tmp_cache
            meetings = self.get_meetings()
            self._cache = old_cache
            if meetings:
                return meetings[0]

        # Fall back to list data
        for meeting in self.get_meetings():
            if meeting.get("id") == meeting_id:
                return meeting

        return None

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information from the underlying source."""
        info = self._source.get_cache_info()

        if self._loaded_at:
            info["last_loaded_ts"] = self._loaded_at.isoformat()

        if self._cache:
            state = self._cache.get("state", {})
            documents = state.get("documents", {})
            info["meeting_count"] = len(documents) if isinstance(documents, dict) else 0
            info["valid_structure"] = True

        return info

    def validate_cache_structure(self) -> bool:
        """Validate that the cache structure is valid."""
        try:
            cache = self.load_cache()
            state = cache.get("state", {})
            return isinstance(state, dict) and "documents" in state
        except Exception:
            return False

    def refresh_cache(self) -> None:
        """Refresh the cache from the source."""
        self._source.refresh_cache()
        self._cache = None
        self._loaded_at = None
