from __future__ import annotations

import logging
from typing import Iterable, Optional

from bootstrap import SRC_DIR  # noqa: F401
from config import EMBEDDING_MODEL, VECTOR_COLLECTION, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)


def _load_embedding_and_chroma():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    return HuggingFaceEmbeddings, Chroma


class TextVectorizer:
    def __init__(self, persist_dir: Optional[str] = None, model_name: str = EMBEDDING_MODEL):
        HuggingFaceEmbeddings, Chroma = _load_embedding_and_chroma()
        self.Chroma = Chroma
        self.persist_dir = str(persist_dir or VECTOR_STORE_PATH)
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def _store(self, collection_name: str = VECTOR_COLLECTION):
        return self.Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
            collection_name=collection_name,
        )

    def build_vector_store(self, documents: Iterable, collection_name: str = VECTOR_COLLECTION, replace: bool = False):
        """Embed chunks. Default is append-by-id so historical PDFs accumulate."""
        docs = list(documents)
        if not docs:
            raise ValueError("No documents to embed")

        ids = []
        for i, doc in enumerate(docs):
            meta = getattr(doc, "metadata", {}) or {}
            source = meta.get("source", "doc")
            chunk_id = meta.get("chunk_id", i)
            ids.append(f"{source}::{chunk_id}")

        if replace:
            vectordb = self.Chroma.from_documents(
                documents=docs,
                embedding=self.embedding_model,
                persist_directory=self.persist_dir,
                collection_name=collection_name,
                ids=ids,
            )
        else:
            vectordb = self._store(collection_name)
            try:
                vectordb.add_documents(docs, ids=ids)
            except Exception as exc:
                logger.warning("Incremental add failed (%s); rebuilding collection", exc)
                vectordb = self.Chroma.from_documents(
                    documents=docs,
                    embedding=self.embedding_model,
                    persist_directory=self.persist_dir,
                    collection_name=collection_name,
                    ids=ids,
                )

        if hasattr(vectordb, "persist"):
            try:
                vectordb.persist()
            except Exception:
                pass
        logger.info("Vector store updated at %s (%s chunks)", self.persist_dir, len(docs))
        return vectordb


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_pipeline.nlp_scraper import RBIScraper

    chunks, _ = RBIScraper().load_and_chunk_pdf("sample_rbi_mpc.pdf")
    TextVectorizer().build_vector_store(chunks)
    print("ChromaDB populated")
