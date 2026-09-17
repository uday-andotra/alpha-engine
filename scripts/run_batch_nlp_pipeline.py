#!/usr/bin/env python3
"""Score every PDF sitting in data/raw/mpc_archive/."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from config import MPC_ARCHIVE_PATH
from data_pipeline.nlp_scraper import RBIScraper
from features.signal_aggregator import SignalAggregator
from features.text_embeddings import TextVectorizer
from models.llm_sentiment import RBISentimentScorer
from models.regime_classifier import RegimeClassifier
from data_pipeline.exports import export_tables

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("archive_nlp")


def list_archive_pdfs(archive_dir: Path) -> list[Path]:
    archive_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in archive_dir.glob("*.pdf") if p.is_file() and p.stat().st_size > 0)


def process_archive(archive_dir: Path | None = None) -> None:
    archive_dir = Path(archive_dir) if archive_dir else MPC_ARCHIVE_PATH
    pdfs = list_archive_pdfs(archive_dir)
    logger.info("Using archive: %s", archive_dir.resolve())
    if not pdfs:
        logger.error("No PDFs found. Copy statements into %s and rerun.", archive_dir.resolve())
        return

    logger.info("Found %s PDF(s): %s", len(pdfs), ", ".join(p.name for p in pdfs))
    logger.info("Loading embedder + Ollama client. Ctrl+C if this sits still more than ~2 minutes.")
    scraper = RBIScraper()
    vectorizer = TextVectorizer()
    scorer = RBISentimentScorer()
    aggregator = SignalAggregator()
    ok = 0

    for pdf_file in pdfs:
        logger.info("Processing %s", pdf_file.name)
        chunks, parsed_date = scraper.load_and_chunk_pdf(pdf_file)
        vectorizer.build_vector_store(chunks, replace=False)
        logger.info("Scoring %s via Ollama (can look idle for up to 90s)...", pdf_file.name)
        result = scorer.score_statement(source_filter=pdf_file.name)
        if not result or "sentiment_score" not in result:
            logger.warning("Skipping %s — LLM did not return a score", pdf_file.name)
            continue
        event_date = result.get("date") or (parsed_date.isoformat() if parsed_date else None)
        if event_date is None:
            logger.warning("Skipping %s — no statement date in filename or text", pdf_file.name)
            continue
        aggregator.fuse_signals(
            latest_sentiment_score=float(result["sentiment_score"]),
            statement_date=event_date,
            source=pdf_file.name,
            reasoning=str(result.get("reasoning") or ""),
        )
        ok += 1
        logger.info("Saved %s -> date=%s score=%s", pdf_file.name, event_date, result["sentiment_score"])

    if ok == 0:
        logger.error("No archive PDF produced a dated score. Check Ollama and filenames.")
        return
    logger.info("Re-fitting regimes from %s scored file(s). This step is silent-ish for 20-40s.", ok)
    RegimeClassifier().fit_markov_switching(causal=True)
    export_tables(("sentiment_events", "fused_features", "regime_predictions"))
    logger.info("Archive NLP complete")


if __name__ == "__main__":
    process_archive()
