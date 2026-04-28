import re
import sys
from pathlib import Path


def _is_underline(line: str, char: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {char}


def extract_changelog_section(changelog: str, version: str) -> list[str]:
    lines = changelog.splitlines()
    heading_pattern = re.compile(rf"^Version {re.escape(version)}(?:\s|\()")

    start = None
    for index, line in enumerate(lines):
        if heading_pattern.match(line):
            start = index
            break

    if start is None:
        raise ValueError(f"Version {version} section not found in docs/changelog.rst")

    end = len(lines)
    for index in range(start + 2, len(lines)):
        if lines[index].startswith("Version ") and index + 1 < len(lines):
            if _is_underline(lines[index + 1], "-"):
                end = index
                break

    return lines[start:end]


def rst_section_to_markdown(lines: list[str]) -> str:
    markdown: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""

        if next_line and _is_underline(next_line, "-"):
            markdown.append(f"## {line.strip()}")
            index += 2
            continue

        if next_line and _is_underline(next_line, "~"):
            markdown.append(f"### {line.strip()}")
            index += 2
            continue

        markdown.append(line.replace("``", "`"))
        index += 1

    return "\n".join(markdown).strip() + "\n"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: scripts/release_notes.py <version>", file=sys.stderr)
        sys.exit(2)

    root = Path(__file__).parent.parent
    changelog_path = root / "docs" / "changelog.rst"

    if not changelog_path.exists():
        print("ERROR: docs/changelog.rst not found", file=sys.stderr)
        sys.exit(1)

    version = (
        sys.argv[1].removeprefix("python-v").removeprefix("julia-v").removeprefix("v")
    )

    try:
        section = extract_changelog_section(changelog_path.read_text(), version)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print(rst_section_to_markdown(section), end="")


if __name__ == "__main__":
    main()
