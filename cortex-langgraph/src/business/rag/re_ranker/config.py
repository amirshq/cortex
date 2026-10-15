from dataclasses import dataclass, field

from src.utils.config import model_settings, rag_settings


def _retrieval(key):
    return field(default_factory=lambda: rag_settings("retrieval")[key])


def _reranker(key):
    return field(default_factory=lambda: rag_settings("reranker")[key])


@dataclass(frozen=True)
class ReRankerConfig:
    """
    Configuration for the re-ranker component in a retrieval-augmented generation system.
    This config is intentionally model-agnostic to allow flexibility in choosing different re-ranking models and
    should not contain any model-specific parameters.

    Every default is read from config.yml when a config is created — nothing is
    hardcoded here any more. Pass a value explicitly to override it for one
    experiment or test, e.g. ReRankerConfig(top_k_input=50).

    Attributes:
        model_name (str): The name of the re-ranking model to be used.        → models.reranker.name
        top_k_input (int): Candidates fetched from the vector DB (recall).    → rag.retrieval.top_k
        top_n_output (int): Chunks kept after re-ranking, sent to the LLM.    → rag.retrieval.top_n
        min_score (float): Relevance gate — chunks below it are dropped.      → rag.reranker.min_score
    """
    # model
    model_name: str = field(default_factory=lambda: model_settings("reranker")["name"])
    device: str = _reranker("device")

    # input/output control
    top_k_input: int = _retrieval("top_k")    # Candidates from vectorDB - control recall
    top_n_output: int = _retrieval("top_n")   # Final results after re-rank and send to PromptBuilder - control precision - directly impact hallucination

    # Gating
    min_score: float = _reranker("min_score")  # filter low-quality candidates and first hallucination firewall

    # Scoring Strategy
    blend_with_vector_score: bool = _reranker("blend_with_vector_score")
    blend_alpha: float = _reranker("blend_alpha")  # α * rerank + (1-α) * vector

    # Performance
    batch_size: int = _reranker("batch_size")
