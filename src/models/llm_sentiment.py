from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

from bootstrap import SRC_DIR  # noqa: F401
from config import EMBEDDING_MODEL, OLLAMA_MODEL, VECTOR_COLLECTION, VECTOR_STORE_PATH

logger = logging.getLogger(__name__)

HAWKISH = (
    "tighten", "tightening", "hike", "hawkish", "inflationary", "vigilant",
    "withdrawal of accommodation", "restrictive", "elevated inflation",
    "upside risk", "food inflation", "core inflation", "anchoring",
)
DOVISH = (
    "dovish", "easing", "rate cut", "accommodative", "support growth",
    "growth impulse", "liquidity", "durable recovery", "neutral to accommodative",
    "policy space", "cut the repo", "reduce the policy",
)

PROMPT = """You are scoring an RBI Monetary Policy Committee statement.

Excerpts:
{context}

Rules:
- +1.0 = strongly hawkish (hikes, fighting inflation, tightening, withdrawal of accommodation)
- -1.0 = strongly dovish (cuts, easing, growth support, accommodation)
- Unchanged repo rate is NOT automatically 0. A hold can still be hawkish or dovish from the language.
- Use 0.0 only if hawkish and dovish evidence is truly balanced.
- Do not copy example numbers. Pick a value in 0.05 steps, e.g. -0.35 or 0.40.

List evidence first, then output JSON only:
{{
  "hawkish_cues": ["short quote", "short quote"],
  "dovish_cues": ["short quote"],
  "sentiment_score": <number between -1 and 1>,
  "date": "YYYY-MM-DD" or null,
  "reasoning": "one sentence"
}}
"""

RETRY_PROMPT = """The previous score was exactly 0.0, which is usually wrong.

Excerpts:
{context}

Force a non-zero tilt unless the text is perfectly two-sided.
JSON only:
{{
  "hawkish_cues": [],
  "dovish_cues": [],
  "sentiment_score": <number>,
  "date": null,
  "reasoning": "one sentence"
}}
"""


def _load_stack():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma
    from langchain_ollama import OllamaLLM
    return HuggingFaceEmbeddings, Chroma, OllamaLLM


def parse_llm_json(response: str) -> Optional[dict]:
    if not response:
        return None
    clean = response.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def keyword_tilt(text: str) -> float:
    blob = (text or "").lower()
    hawk = sum(1 for w in HAWKISH if w in blob)
    dove = sum(1 for w in DOVISH if w in blob)
    total = hawk + dove
    if total == 0:
        return 0.0
    raw = (hawk - dove) / total
    return float(max(-0.6, min(0.6, raw)))


def ping_ollama(host: str = "http://127.0.0.1:11434", timeout: int = 5) -> bool:
    try:
        import requests
        res = requests.get(f"{host}/api/tags", timeout=timeout)
        return res.status_code == 200
    except Exception as exc:
        logger.warning("Ollama is not reachable at %s (%s)", host, exc)
        return False


class RBISentimentScorer:
    def __init__(self, model_name: str = OLLAMA_MODEL):
        HuggingFaceEmbeddings, Chroma, OllamaLLM = _load_stack()
        self.model_name = model_name
        self.host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        self.timeout = int(os.getenv("ALPHA_OLLAMA_TIMEOUT", "90"))
        self.persist_dir = str(VECTOR_STORE_PATH)
        logger.info("Loading embedding model %s (first time can take a few minutes)", EMBEDDING_MODEL)
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = None
        self.vector_store = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_model,
            collection_name=VECTOR_COLLECTION,
        )
        if ping_ollama(self.host):
            logger.info("Ollama is up at %s", self.host)
        else:
            logger.warning("Ollama ping failed; scoring will fall back to keywords if generate times out")

    def _ask(self, prompt: str) -> Optional[dict]:
        import requests

        logger.info("Calling Ollama model %s (timeout %ss). First load of the model can look frozen.", self.model_name, self.timeout)
        started = time.time()
        try:
            res = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 256},
                },
                timeout=self.timeout,
            )
            res.raise_for_status()
            response = res.json().get("response", "")
        except Exception as exc:
            logger.error("Ollama generate failed after %.1fs: %s", time.time() - started, exc)
            return None
        logger.info("Ollama responded in %.1fs", time.time() - started)
        payload = parse_llm_json(response)
        if payload is None:
            logger.error("Failed to parse JSON. Raw output:\n%s", response[:1000])
            return None
        try:
            payload["sentiment_score"] = float(payload.get("sentiment_score"))
        except (TypeError, ValueError):
            logger.error("sentiment_score missing or invalid: %s", payload)
            return None
        payload["sentiment_score"] = max(-1.0, min(1.0, payload["sentiment_score"]))
        return payload

    def score_statement(
        self,
        query: str = "repo rate inflation growth stance hike cut accommodation",
        source_filter: Optional[str] = None,
    ):
        logger.info("Retrieving context from vector store...")
        try:
            if source_filter:
                docs = self.vector_store.similarity_search(query, k=8, filter={"source": source_filter})
            else:
                docs = self.vector_store.similarity_search(query, k=8)
        except Exception as exc:
            logger.error("Vector store query failed: %s", exc)
            return None
        if not docs:
            logger.warning("No documents in vector store")
            return None

        context_text = "\n\n".join(doc.page_content[:800] for doc in docs[:4])
        meta_date = None
        for doc in docs:
            meta_date = (doc.metadata or {}).get("statement_date") or meta_date

        payload = self._ask(PROMPT.format(context=context_text))
        if payload is None:
            tilt = keyword_tilt(context_text)
            logger.warning("LLM unavailable; using keyword tilt %.2f", tilt)
            return {"sentiment_score": tilt, "date": meta_date, "reasoning": "Keyword fallback (Ollama timed out or failed).", "hawkish_cues": [], "dovish_cues": []}

        if abs(payload["sentiment_score"]) < 1e-9 and os.getenv("ALPHA_LLM_RETRY", "0") == "1":
            logger.info("Got exactly 0.0; retrying with a stricter prompt")
            retry = self._ask(RETRY_PROMPT.format(context=context_text))
            if retry is not None:
                payload = retry

        tilt = keyword_tilt(context_text)
        if abs(payload["sentiment_score"]) < 0.05 and abs(tilt) >= 0.15:
            logger.info("LLM score near 0; applying keyword tilt %.2f", tilt)
            payload["sentiment_score"] = tilt
            payload["reasoning"] = (
                f"{payload.get('reasoning', '')} Keyword tilt applied because the model returned ~0."
            ).strip()

        if not payload.get("date"):
            payload["date"] = meta_date
        logger.info(
            "Sentiment=%s date=%s hawkish=%s dovish=%s (%s)",
            payload["sentiment_score"],
            payload.get("date"),
            payload.get("hawkish_cues"),
            payload.get("dovish_cues"),
            payload.get("reasoning"),
        )
        return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(RBISentimentScorer().score_statement())
