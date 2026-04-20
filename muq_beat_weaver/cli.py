from __future__ import annotations

import argparse
from pathlib import Path


def _should_process_map_folder(map_folder: Path, input_root: Path) -> bool:
    try:
        rel_parts = [p.lower() for p in Path(map_folder).relative_to(Path(input_root)).parts]
    except ValueError:
        rel_parts = [p.lower() for p in Path(map_folder).parts]
    ignored_parts = {"autosaves", "backups"}
    return not any(part in ignored_parts or part.startswith("~") for part in rel_parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="muq-beat-weaver")
    subparsers = parser.add_subparsers(dest="command")
    for name in ("train", "generate", "evaluate", "embed-muq"):
        subparsers.add_parser(name)
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()

