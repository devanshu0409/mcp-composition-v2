"""Semantic tool index: cosine similarity over tool embeddings.

Backend priority:
  1. sentence-transformers (all-MiniLM-L6-v2) — best quality
  2. sklearn TF-IDF — fast, offline fallback
  3. substring match — zero-dep last resort
"""
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolRecord:
    namespace: str
    original_name: str
    qualified_name: str
    description: str
    input_schema: dict[str, Any]
    embedding_text: str = field(default="", init=False)


def _build_embedding_text(tool: ToolRecord) -> str:
    """Combine name, description, and schema property names for richer signal."""
    parts = [tool.original_name.replace("_", " ")]
    if tool.description:
        parts.append(tool.description)
    props = tool.input_schema.get("properties", {})
    for prop_name, prop_schema in props.items():
        prop_desc = prop_schema.get("description", "")
        parts.append(f"{prop_name.replace('_', ' ')}: {prop_desc}" if prop_desc else prop_name.replace("_", " "))
    return " | ".join(parts)


def _detect_backend() -> str:
    try:
        import sentence_transformers  # noqa: F401
        return "sentence_transformers"
    except ImportError:
        pass
    try:
        import sklearn  # noqa: F401
        return "tfidf"
    except ImportError:
        pass
    return "substring"


class SemanticIndex:
    def __init__(self) -> None:
        self._tools: list[ToolRecord] = []
        self._embeddings: Any = None
        self._backend: str = "none"
        self._model: Any = None
        self._is_built: bool = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    async def build(self, tools: list[ToolRecord]) -> None:
        """Build the semantic index from tool records. Safe to call from async context."""
        for tool in tools:
            tool.embedding_text = _build_embedding_text(tool)

        self._tools = tools
        self._backend = _detect_backend()
        texts = [t.embedding_text for t in tools]

        logger.info("Building semantic index: %d tools, backend=%s", len(tools), self._backend)

        loop = asyncio.get_running_loop()
        if self._backend == "sentence_transformers":
            try:
                self._embeddings = await loop.run_in_executor(None, self._build_st, texts)
            except Exception as exc:
                logger.warning("sentence-transformers failed (%s), falling back to TF-IDF", exc)
                self._backend = "tfidf"
                self._embeddings = await loop.run_in_executor(None, self._build_tfidf, texts)
        elif self._backend == "tfidf":
            self._embeddings = await loop.run_in_executor(None, self._build_tfidf, texts)
        # substring: no precomputed embeddings needed

        self._is_built = True
        logger.info("Semantic index ready (backend=%s)", self._backend)

    def _build_st(self, texts: list[str]) -> Any:
        from sentence_transformers import SentenceTransformer
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _build_tfidf(self, texts: list[str]) -> Any:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        self._model = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        matrix = self._model.fit_transform(texts)
        return normalize(matrix, norm="l2")

    async def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top_k tools by cosine similarity."""
        if not self._is_built or not self._tools:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query, top_k)

    def _search_sync(self, query: str, top_k: int) -> list[dict[str, Any]]:
        import numpy as np

        n = min(top_k, len(self._tools))

        if self._backend == "sentence_transformers":
            q_emb = self._model.encode([query], normalize_embeddings=True)[0]
            scores = self._embeddings @ q_emb

        elif self._backend == "tfidf":
            from sklearn.preprocessing import normalize as sk_norm
            q_vec = self._model.transform([query])
            q_vec = sk_norm(q_vec, norm="l2")
            scores = (self._embeddings @ q_vec.T).toarray().flatten()

        else:
            q_lower = query.lower()
            scores = np.array([
                sum(1.0 for word in q_lower.split() if word in t.embedding_text.lower()) /
                max(len(q_lower.split()), 1)
                for t in self._tools
            ])

        top_indices = np.argsort(scores)[::-1][:n]
        results = []
        for idx in top_indices:
            tool = self._tools[int(idx)]
            results.append({
                "tool": tool.qualified_name,
                "score": round(float(scores[idx]), 4),
                "description": tool.description,
                "schema": tool.input_schema,
            })
        return results
