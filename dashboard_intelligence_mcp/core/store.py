"""Reference archive storage.

The store loads a read-only seed archive shipped with the package and merges it
with a writable user archive (references registered at runtime). Persistence is
a plain JSON file so the server has zero external database dependency for the
MVP; the schema mirrors the ``references`` + ``visual_analysis`` tables from the
design spec and can be migrated to PostgreSQL + a vector DB later without
changing the tool contracts.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

_PKG_DIR = Path(__file__).resolve().parent.parent
_SEED_PATH = _PKG_DIR / "data" / "references.seed.json"

# Fields that make up a reference record. ``visual_analysis`` is nested.
REFERENCE_FIELDS = [
    "id",
    "title",
    "source_url",
    "source_platform",
    "creator",
    "thumbnail_url",
    "original_image_url",
    "industry",
    "dashboard_type",
    "primary_user",
    "business_goal",
    "rights_status",
    "tags",
    "supported_metrics",
    "decision_purpose",
    "visual_analysis",
]


def _user_store_path() -> Path:
    """Location of the writable user archive.

    Honors ``DASHBOARD_MCP_DATA_DIR`` so a deployment can point at a persistent
    volume; otherwise writes next to the seed file.
    """
    override = os.getenv("DASHBOARD_MCP_DATA_DIR")
    base = Path(override) if override else (_PKG_DIR / "data")
    base.mkdir(parents=True, exist_ok=True)
    return base / "references.user.json"


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug or "reference"


class ReferenceStore:
    """In-memory archive backed by seed + user JSON files."""

    def __init__(
        self,
        seed_path: Optional[Path] = None,
        user_path: Optional[Path] = None,
    ) -> None:
        self._seed_path = Path(seed_path) if seed_path else _SEED_PATH
        self._user_path = Path(user_path) if user_path else _user_store_path()
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self.reload()

    # ---- loading -------------------------------------------------------
    def reload(self) -> None:
        self._by_id = {}
        for record in self._read(self._seed_path):
            self._by_id[record["id"]] = record
        for record in self._read(self._user_path):
            # User records override seed records with the same id.
            self._by_id[record["id"]] = record

    @staticmethod
    def _read(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as fh:
                text = fh.read().strip()
            if not text:
                return []
            payload = json.loads(text)
        except (json.JSONDecodeError, OSError):
            # A corrupt/empty user archive must not crash the server.
            return []
        return payload.get("references", [])

    def _write_user(self) -> None:
        user_records = [
            r for r in self._by_id.values() if r.get("_origin") == "user"
        ]
        payload = {"schema_version": 1, "references": user_records}
        with self._user_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---- reads ---------------------------------------------------------
    def all(self) -> List[Dict[str, Any]]:
        return [deepcopy(r) for r in self._by_id.values()]

    def get(self, reference_id: str) -> Optional[Dict[str, Any]]:
        record = self._by_id.get(reference_id)
        return deepcopy(record) if record else None

    def by_industry(self, industry: str) -> List[Dict[str, Any]]:
        industry = (industry or "").lower()
        return [
            deepcopy(r)
            for r in self._by_id.values()
            if (r.get("industry") or "").lower() == industry
        ]

    def industries(self) -> List[str]:
        return sorted({r.get("industry") for r in self._by_id.values() if r.get("industry")})

    def patterns(self) -> List[str]:
        seen = {}
        for r in self._by_id.values():
            pattern = (r.get("visual_analysis") or {}).get("layout_pattern")
            if pattern:
                seen.setdefault(_slugify(pattern), pattern)
        return sorted(seen)

    def by_pattern_slug(self, slug: str) -> List[Dict[str, Any]]:
        slug = _slugify(slug)
        out = []
        for r in self._by_id.values():
            pattern = (r.get("visual_analysis") or {}).get("layout_pattern")
            if pattern and _slugify(pattern) == slug:
                out.append(deepcopy(r))
        return out

    def count(self) -> int:
        return len(self._by_id)

    # ---- writes --------------------------------------------------------
    def register(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Validate, normalize and persist a new reference.

        Returns the stored record. Raises ``ValueError`` on missing required
        fields.
        """
        record = deepcopy(record)
        title = record.get("title")
        if not title:
            raise ValueError("`title` is required to register a reference")
        if not record.get("source_url") and not record.get("original_image_url"):
            raise ValueError(
                "at least one of `source_url` or `original_image_url` is required"
            )

        ref_id = record.get("id") or self._next_id(title)
        if ref_id in self._by_id and self._by_id[ref_id].get("_origin") != "user":
            # Never silently overwrite a shipped seed reference.
            ref_id = self._next_id(title)
        record["id"] = ref_id

        record.setdefault("tags", [])
        record.setdefault("supported_metrics", [])
        record.setdefault("decision_purpose", [])
        record.setdefault("rights_status", "reference_only")
        record.setdefault("visual_analysis", {})
        record["_origin"] = "user"

        self._by_id[ref_id] = record
        self._write_user()
        return deepcopy(record)

    def _next_id(self, title: str) -> str:
        base = f"REF-{_slugify(title)[:24]}"
        candidate = base
        n = 2
        while candidate in self._by_id:
            candidate = f"{base}-{n}"
            n += 1
        return candidate
