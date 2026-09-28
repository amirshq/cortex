from dataclasses import dataclass

@dataclass(frozen=True)
class ReRankerConfig:
    """
    Configuration for the re-ranker component in a retrieval-augmented generation system.
    This config is intentionally model-agnostic to allow flexibility in choosing different re-ranking models and
    should not contain any model-specific parameters.
    
    Attributes:
        model_name (str): The name of the re-ranking model to be used.
        top_k (int): The number of top documents to consider for re-ranking.
        threshold (float): The score threshold for filtering documents after re-ranking.
    """
    #model
    model_name: str = 'BAAI/bge-reranker-base'
    device: str = 'cpu'

    # input/output control
    top_k_input: int = 30   #Candidates from vectorDB - control recall
    top_n_output: int = 8   #Final results after re-rank and send to PromptBuilder - control precision - directly impact hallucination

    # Gating 
    min_score: float = 0.15 # filter low-quality candidates and first hallucination firewall

    # Scoring Strategy
    blend_with_vector_score: bool = False
    blend_alpha: float = 0.5        # α * rerank + (1-α) * vector

    # Performance
    batch_size: int = 8