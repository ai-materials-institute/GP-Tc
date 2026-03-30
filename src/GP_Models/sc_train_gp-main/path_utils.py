from pathlib import Path


SC_TRAIN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SC_TRAIN_ROOT.parents[2]
REPLICATION_DATA_ROOT = PROJECT_ROOT / "data" / "replication"


def resolve_replication_data_file(*relative_parts: str) -> Path:
    relative_path = Path(*relative_parts)
    candidates = [
        REPLICATION_DATA_ROOT / relative_path,
        SC_TRAIN_ROOT / relative_path,
        SC_TRAIN_ROOT / "replication" / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
