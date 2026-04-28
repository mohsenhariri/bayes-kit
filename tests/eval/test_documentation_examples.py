from __future__ import annotations

import contextlib
import io
import pathlib
import re
import textwrap


ROOT = pathlib.Path(__file__).resolve().parents[2]

PYTHON_EXAMPLE_FILES = [
    pathlib.Path("docs/index.rst"),
    pathlib.Path("docs/quickstart.rst"),
    pathlib.Path("docs/examples.rst"),
    pathlib.Path("README.md"),
    pathlib.Path("README_PyPI.md"),
]


def _markdown_python_blocks(text: str) -> list[str]:
    return [
        textwrap.dedent(match.group(1))
        for match in re.finditer(r"```python\n(.*?)\n```", text, flags=re.S)
    ]


def _rst_python_blocks(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    i = 0

    while i < len(lines):
        if lines[i].strip() != ".. code-block:: python":
            i += 1
            continue

        i += 1
        while i < len(lines) and (not lines[i].strip() or lines[i].startswith("   ")):
            if lines[i].startswith("   "):
                break
            i += 1

        block: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.startswith("   "):
                block.append(line[3:])
                i += 1
                continue
            if not line.strip():
                block.append("")
                i += 1
                continue
            break

        blocks.append("\n".join(block).rstrip() + "\n")

    return blocks


def _python_blocks(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".md":
        return _markdown_python_blocks(text)
    return _rst_python_blocks(text)


def test_user_facing_python_examples_execute() -> None:
    for relative_path in PYTHON_EXAMPLE_FILES:
        path = ROOT / relative_path
        namespace = {"__name__": f"__doc_examples_{relative_path.as_posix()}__"}

        for idx, code in enumerate(_python_blocks(path), start=1):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, f"{relative_path}:block{idx}", "exec"), namespace)
