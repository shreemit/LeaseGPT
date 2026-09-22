#!/usr/bin/env python3
"""Push HEAD to a Hugging Face Space with card metadata on README.md.

Not imported by the app. Hugging Face reads ``sdk`` and ``app_port`` from YAML
at the top of the Space README. That block stays in ``huggingface/space.yml``
so the GitHub README opens on the project title.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "huggingface" / "space.yml"


def with_space_card(readme: str, metadata: str) -> str:
    if readme.startswith("---\n") or readme.startswith("---\r\n"):
        return readme
    meta = metadata.strip() + "\n"
    body = readme.lstrip("\n")
    return f"---\n{meta}---\n\n{body}"


def main(argv: list[str]) -> int:
    if len(argv) != 2 or not argv[1].startswith("https://huggingface.co/spaces/"):
        print(
            "usage: uv run python scripts/push_huggingface_space.py "
            "https://huggingface.co/spaces/<user>/<space>",
            file=sys.stderr,
        )
        return 2
    url = argv[1]
    if not METADATA.is_file():
        print(f"missing {METADATA}", file=sys.stderr)
        return 1

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        print(
            "Working tree has uncommitted changes. This push uses the current HEAD only.",
            file=sys.stderr,
        )

    metadata = METADATA.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "repo"
        subprocess.run(
            ["git", "clone", "--local", "--no-hardlinks", str(ROOT), str(clone)],
            check=True,
        )
        readme_path = clone / "README.md"
        original = readme_path.read_text(encoding="utf-8")
        updated = with_space_card(original, metadata)
        if updated != original:
            readme_path.write_text(updated, encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=clone, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=LeaseGPT",
                    "-c",
                    "user.email=leasegpt@users.noreply.github.com",
                    "commit",
                    "-m",
                    "Prepend Hugging Face Space card metadata",
                ],
                cwd=clone,
                check=True,
            )
        subprocess.run(["git", "push", url, "HEAD:main"], cwd=clone, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
