#!/usr/bin/env python3
"""Score PDFs in the archive, or one file if you pass a path."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import MPC_ARCHIVE_PATH

from run_batch_nlp_pipeline import process_archive

if __name__ == "__main__":
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
        dest = MPC_ARCHIVE_PATH / src.name
        MPC_ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        print(f"Using {dest}")
    process_archive()
