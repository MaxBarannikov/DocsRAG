"""The Makefile must not export .env into child processes.

Make keeps quotes as part of the value, and python-dotenv will not override a variable
already in the environment — which is how valid LangFuse keys were rejected with a 401.
"""

from __future__ import annotations

import re
from pathlib import Path

MAKEFILE = Path("Makefile")


def test_makefile_does_not_export_dotenv() -> None:
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    include_line = next((i for i, line in enumerate(lines) if re.match(r"\s*include\s+\.env", line)), None)
    assert include_line is not None, "the Makefile should still include .env for $(VAR) interpolation"

    bare_exports = [line for line in lines if re.match(r"^\s*export\s*$", line)]
    assert not bare_exports, "a bare `export` re-introduces the quoted-value bug"


def test_env_example_values_are_unquoted() -> None:
    """Legal for dotenv, but invites the Make mismatch."""
    for raw in Path(".env.example").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        _, _, value = line.partition("=")
        assert not value.startswith(('"', "'")), f"quoted value in .env.example: {line}"
