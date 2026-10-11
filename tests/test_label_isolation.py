"""Hard constraint 3: only ``eval/`` may see evaluation labels.

A static scan, so it holds for code paths no test happens to execute. Outside
``src/policygate_rag/eval/`` no module may

* import ``policygate_rag.eval`` (absolutely or relatively);
* name an evaluation field -- ``required_documents``, ``evaluation_criteria``
  and the rest -- as an attribute, a string or an identifier. tau2's ``Task``
  carries ``required_documents`` as a plain attribute, so holding a task object
  is enough to leak the labels; reading the task files directly is the other
  route;
* name a task file.

A false positive (say, a docstring mentioning the field) is fixed by rewording,
not by an allowlist: the cost of rewording is a minute, and an allowlist is
where leaks would hide. The scanner is itself tested below against planted
violations, so a scan that silently matches nothing cannot pass.
"""

import ast
import io
import tokenize
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "policygate_rag"
EVAL = SRC / "eval"

LABEL_FIELDS = (
    "required_documents",
    "evaluation_criteria",
    "reward_basis",
    "nl_assertions",
    "communicate_info",
    "env_assertions",
)
TASK_FILES = ("tasks.json", "tasks_voice.json")


def violations(source: str, *, path: str = "<string>") -> list[str]:
    """Every way ``source`` reaches evaluation labels, as human-readable lines."""
    found: list[str] = []
    tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "policygate_rag.eval" or alias.name.startswith(
                    "policygate_rag.eval."
                ):
                    found.append(f"{path}:{node.lineno}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            absolute = module == "policygate_rag.eval" or module.startswith("policygate_rag.eval.")
            relative = node.level > 0 and (module == "eval" or module.startswith("eval."))
            from_parent = node.level > 0 and module in ("", "policygate_rag") and any(
                a.name == "eval" for a in node.names
            )
            if absolute or relative or from_parent:
                found.append(f"{path}:{node.lineno}: imports eval ({'.' * node.level}{module})")
            if module == "policygate_rag" and any(a.name == "eval" for a in node.names):
                found.append(f"{path}:{node.lineno}: imports policygate_rag.eval")

    # Tokens rather than the AST for names: this also catches getattr(t, "...")
    # and f-strings, which an attribute-only check would miss.
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in (tokenize.NAME, tokenize.STRING, tokenize.FSTRING_MIDDLE):
            continue
        for field in LABEL_FIELDS:
            if field in tok.string:
                found.append(f"{path}:{tok.start[0]}: names label field {field!r}")
        if tok.type != tokenize.NAME:
            for name in TASK_FILES:
                if name in tok.string:
                    found.append(f"{path}:{tok.start[0]}: names task file {name!r}")
    return found


def scanned_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if EVAL not in p.parents)


def test_the_scan_covers_the_package():
    files = scanned_files()
    assert files, "nothing scanned: the package layout moved and this test went blind"
    assert all(EVAL not in p.parents for p in files)


def test_no_module_outside_eval_reaches_labels():
    problems = [
        line
        for path in scanned_files()
        for line in violations(path.read_text(), path=str(path.relative_to(SRC.parent)))
    ]
    assert not problems, "label leak outside eval/:\n" + "\n".join(problems)


# --- the scanner catches what it claims to -------------------------------------

@pytest.mark.parametrize(
    "source",
    [
        "import policygate_rag.eval\n",
        "import policygate_rag.eval.metrics\n",
        "from policygate_rag.eval import metrics\n",
        "from policygate_rag.eval.metrics import coverage\n",
        "from policygate_rag import eval\n",
        "from ..eval import metrics\n",
        "from ..eval.metrics import coverage\n",
        "from .. import eval\n",
        "docs = task.required_documents\n",
        "docs = getattr(task, 'required_documents')\n",
        "key = f'{prefix}required_documents'\n",
        "crit = task.evaluation_criteria\n",
        "path = DATA / 'tasks.json'\n",
    ],
)
def test_the_scanner_flags_planted_violations(source):
    assert violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "from policygate_rag.index import bm25\n",
        "from ..retrieval import naive\n",
        "import policygate.harness.session\n",
        "x = evaluate(documents)\n",
    ],
)
def test_the_scanner_passes_clean_code(source):
    assert violations(source) == []
