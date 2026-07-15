"""Pure-Python core for Dashboard Intelligence MCP (no MCP dependency)."""

from .store import ReferenceStore
from .scoring import score_reference, search_references
from .analysis import analyze_reference, compare_references
from .blueprint import generate_blueprint, generate_build_prompt
from .audit import audit_dashboard

__all__ = [
    "ReferenceStore",
    "score_reference",
    "search_references",
    "analyze_reference",
    "compare_references",
    "generate_blueprint",
    "generate_build_prompt",
    "audit_dashboard",
]
