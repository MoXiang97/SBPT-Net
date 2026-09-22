"""Check that the public release tree contains no private or oversized artifacts."""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List

BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".onnx", ".pkl", ".pickle"}
SECRET_NAMES = {".env", "credentials.json", "secrets.json", "id_rsa", "id_ed25519"}
SKIP_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".superpowers",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}
MAX_FILE_BYTES = 20 * 1024 * 1024
WINDOWS_ABSOLUTE = re.compile(
    r"[A-Za-z]:\\(?:Users|Data|Program Files|Windows)\\", re.IGNORECASE
)


def verify_release_tree(root: str | Path) -> List[str]:
    root = Path(root).resolve()
    errors: List[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if path.name.lower() in SECRET_NAMES or path.suffix.lower() in BINARY_SUFFIXES:
            errors.append(f"forbidden artifact: {relative.as_posix()}")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"oversized file: {relative.as_posix()}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if WINDOWS_ABSOLUTE.search(text):
            errors.append(f"local absolute path: {relative.as_posix()}")
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    errors = verify_release_tree(args.root)
    if args.checkpoint:
        from s2rspc.config import load_config
        from s2rspc.engine.checkpoint import load_checkpoint

        cfg = load_config(Path("configs/sbptnet_sts2r.yaml"))
        load_checkpoint(args.checkpoint, cfg.model)
        print(f"checkpoint compatible: {args.checkpoint}")
    if errors:
        for error in errors:
            print(error)
        return 1
    print("release tree: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
