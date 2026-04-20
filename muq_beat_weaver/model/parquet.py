from __future__ import annotations

from pathlib import Path

def read_notes_parquet(path: Path):
    """Read notes Parquet file(s) from a processed-data directory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("notes_*.parquet"))
        if not files:
            single = path / "notes.parquet"
            if single.exists():
                return pq.read_table(single)
            raise FileNotFoundError(f"No notes Parquet files in {path}")
        return pa.concat_tables([pq.read_table(file) for file in files])
    return pq.read_table(path)
