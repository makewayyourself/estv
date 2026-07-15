"""Dashboard Intelligence MCP.

A Model Context Protocol server that turns a dashboard-reference archive into a
design-decision engine: search references by business intent, analyze their
information hierarchy, compare candidates, generate a data-mapped blueprint,
and emit build prompts for AI coding tools.

The ``core`` subpackage is pure Python with no MCP dependency so it can be unit
tested and reused directly. ``server`` is a thin MCP wrapper over ``core``.
"""

from .core.store import ReferenceStore  # noqa: F401

__all__ = ["ReferenceStore"]
__version__ = "0.1.0"
