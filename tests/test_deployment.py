"""Deployment-readiness checks for Streamlit Community Cloud.

These tests guard the properties a clean, remote deployment depends on:

1. Every third-party package the code imports is declared in
   ``requirements.txt`` (Community Cloud installs exactly that file - an
   undeclared import would crash the deployed app).
2. Every source file parses with the Python 3.12 grammar (the Community
   Cloud default interpreter), so no newer-syntax-only construct sneaks in
   while local development runs on a newer Python.
3. ``.gitignore`` keeps caches, virtual environments and secret files out
   of Git so they can never be published with the repository.
4. User-facing repository files contain no absolute personal paths.

Nothing here touches the network; all checks read local files.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Import name -> distribution name where they differ.
IMPORT_TO_REQUIREMENT = {"serial": "pyserial"}

#: Local modules (one per root-level ``*.py``) that are not packages.
LOCAL_MODULES = {path.stem for path in ROOT.glob("*.py")}


def _python_sources() -> list[Path]:
    return sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))


def _declared_requirements() -> set[str]:
    names: set[str] = set()
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def _third_party_imports() -> set[str]:
    found: set[str] = set()
    for path in _python_sources():
        tree = ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level or node.module is None:
                    continue
                modules = [node.module.split(".")[0]]
            else:
                continue
            for module in modules:
                if module in sys.stdlib_module_names:
                    continue
                if module in LOCAL_MODULES or module == "tests":
                    continue
                found.add(IMPORT_TO_REQUIREMENT.get(module, module))
    return found


def test_third_party_imports_are_declared_in_requirements():
    """Community Cloud installs only requirements.txt - keep it complete."""
    used = _third_party_imports()
    declared = _declared_requirements()
    missing = used - declared
    assert not missing, (
        f"imports used by the code but missing from requirements.txt: "
        f"{sorted(missing)}"
    )
    # Sanity: the well-known names are present, so a parsing mistake in the
    # helpers above cannot silently make this test vacuous.
    assert {"streamlit", "pandas", "pyserial"} <= declared


def test_sources_parse_with_python_312_grammar():
    """Community Cloud defaults to Python 3.12 - reject newer-only syntax."""
    offenders: list[str] = []
    for path in _python_sources():
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(
                source,
                filename=str(path),
                feature_version=(3, 12),
            )
        except SyntaxError as exc:
            offenders.append(f"{path.name}: {exc}")
    assert not offenders, (
        "files that use syntax unavailable on Python 3.12: "
        + "; ".join(offenders)
    )


def test_gitignore_excludes_secrets_caches_and_test_output():
    """Nothing local or private may ever be committed by accident."""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required_patterns = (
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        ".streamlit/secrets.toml",
        ".env",
        "pytest_*.txt",
        ".venv/",
    )
    for pattern in required_patterns:
        assert pattern in text, f".gitignore no longer excludes {pattern!r}"


def test_repository_files_contain_no_absolute_personal_paths():
    """README/requirements must stay portable and free of local paths."""
    for filename in ("README.md", "requirements.txt"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        for pattern in ("C:\\Users", "C:/Users", "/home/", "/Users/"):
            assert pattern not in text, (
                f"{filename} contains a personal path pattern {pattern!r}"
            )
