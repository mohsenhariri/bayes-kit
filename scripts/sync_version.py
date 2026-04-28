import re
import sys
from pathlib import Path


def update_file(root, relative_path, replacements):
    path = root / relative_path
    if not path.exists():
        print(f"WARNING: {relative_path} not found")
        return

    content = path.read_text()
    new_content = content

    for pattern, replacement in replacements:
        new_content, count = re.subn(pattern, replacement, new_content)
        if count == 0:
            print(f"WARNING: pattern not found in {relative_path}: {pattern}")

    if new_content != content:
        path.write_text(new_content)

    print(f"Updated {relative_path}")


def main():
    """Read VERSION file and update all version references."""
    root = Path(__file__).parent.parent
    version_file = root / "VERSION"

    if not version_file.exists():
        print("ERROR: VERSION file not found")
        sys.exit(1)

    version = version_file.read_text().strip()

    if not version:
        print("ERROR: VERSION file is empty")
        sys.exit(1)

    print(f"Syncing to version: {version}")

    update_file(
        root,
        Path("scorio/__init__.py"),
        [(r'__version__ = "[^"]*"', f'__version__ = "{version}"')],
    )
    update_file(
        root,
        Path("julia/Scorio.jl/Project.toml"),
        [(r'(?m)^version = "[^"]*"', f'version = "{version}"')],
    )
    update_file(
        root,
        Path("julia/Scorio.jl/src/Scorio.jl"),
        [(r'const VERSION = v"[^"]*"', f'const VERSION = v"{version}"')],
    )
    update_file(
        root,
        Path("julia/Scorio.jl/test/runtests.jl"),
        [(r'Scorio\.VERSION == v"[^"]*"', f'Scorio.VERSION == v"{version}"')],
    )
    update_file(
        root,
        Path("julia/Scorio.jl/docs/Manifest.toml"),
        [
            (
                r'(?ms)(\[\[deps\.Scorio\]\].*?^version = ")[^"]*(")',
                rf"\g<1>{version}\2",
            )
        ],
    )
    update_file(
        root,
        Path("docs/conf.py"),
        [(r'(?m)^release = "[^"]*"', f'release = "{version}"')],
    )
    update_file(
        root,
        Path("CITATION.cff"),
        [(r"(?m)^version: .*$", f"version: {version}")],
    )
    tag_replacement = [(r"scorio\.git@v\d+\.\d+\.\d+", f"scorio.git@v{version}")]
    update_file(root, Path("README.md"), tag_replacement)
    update_file(root, Path("README_PyPI.md"), tag_replacement)

    print(f"\nAll versions synced to {version}")


if __name__ == "__main__":
    main()
