"""Hard constraint 7: evaluation code is independent of the system under test.

``eval/``, ``index/`` and ``judges/`` are pure Python: no LangGraph, LangChain
or LLM-client imports. The API-backed judges' thin client wrappers are the one
exception the constraint names; they must live in a module whose name ends in
``_client.py`` so the exemption is visible in the file tree, not hidden in a
list here.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "policygate_rag"
PURE = ("eval", "index", "judges")
FORBIDDEN = ("langgraph", "langchain", "langchain_core", "litellm", "openai")


def framework_imports(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        for name in names:
            if name.split(".")[0] in FORBIDDEN:
                found.append(f"{node.lineno}: {name}")
    return found


def test_pure_packages_import_no_framework_or_llm_client():
    problems = []
    for package in PURE:
        root = SRC / package
        assert root.is_dir(), f"{root} missing: this test would scan nothing"
        for path in sorted(root.rglob("*.py")):
            if path.name.endswith("_client.py"):
                continue
            rel = path.relative_to(SRC)
            problems += [f"{rel}:{p}" for p in framework_imports(path.read_text())]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "source",
    ["import langgraph\n", "from langgraph.graph import StateGraph\n", "import openai\n",
     "from litellm import completion\n"],
)
def test_the_check_flags_planted_imports(source):
    assert framework_imports(source)
