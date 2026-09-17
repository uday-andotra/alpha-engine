#!/usr/bin/env python3
"""No automatic download. This only shows the manual archive folder."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from config import MPC_ARCHIVE_PATH

archive = MPC_ARCHIVE_PATH
archive.mkdir(parents=True, exist_ok=True)
pdfs = sorted(archive.glob("*.pdf"))

print("Automatic RBI download is disabled.")
print("Put Monetary Policy PDFs here:")
print(f"  {archive.resolve()}")
print()
if pdfs:
    print(f"{len(pdfs)} file(s) already in the archive:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")
    print()
    print("Score them with:")
    print("  python scripts/run_batch_nlp_pipeline.py")
else:
    print("Archive is empty. Add .pdf files, then run:")
    print("  python scripts/run_batch_nlp_pipeline.py")
