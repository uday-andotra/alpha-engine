from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from bootstrap import SRC_DIR  # noqa: F401
from config import CHUNK_OVERLAP, CHUNK_SIZE, RAW_DATA_PATH

from data_pipeline.dates import parse_date_from_filename, parse_date_from_text

logger = logging.getLogger(__name__)


class RBIScraper:
    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = Path(raw_dir) if raw_dir else RAW_DATA_PATH
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )

    def resolve_pdf(self, filename: Union[str, Path]) -> Path:
        candidate = Path(filename)
        if candidate.is_file():
            return candidate
        nested = self.raw_dir / candidate
        if nested.is_file():
            return nested
        archive = self.raw_dir / "mpc_archive" / candidate.name
        if archive.is_file():
            return archive
        raise FileNotFoundError(f"PDF not found: {filename} (looked in {self.raw_dir})")

    def load_and_chunk_pdf(self, filename: Union[str, Path]):
        full_path = self.resolve_pdf(filename)
        loader = PyPDFLoader(str(full_path))
        raw_pages = loader.load()
        chunks = self.text_splitter.split_documents(raw_pages)
        statement_date = parse_date_from_filename(full_path.name)
        if statement_date is None:
            preview = " ".join(p.page_content[:800] for p in raw_pages[:2])
            statement_date = parse_date_from_text(preview)
        for i, chunk in enumerate(chunks):
            chunk.metadata = dict(chunk.metadata or {})
            chunk.metadata["source"] = str(full_path.name)
            chunk.metadata["chunk_id"] = i
            if statement_date:
                chunk.metadata["statement_date"] = statement_date.isoformat()
        logger.info("Chunked %s into %s segments (date=%s)", full_path.name, len(chunks), statement_date)
        return chunks, statement_date


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = RBIScraper()
    chunks, dt = scraper.load_and_chunk_pdf("sample_rbi_mpc.pdf")
    print(f"date={dt} n_chunks={len(chunks)}")
    print(chunks[0].page_content[:200])
