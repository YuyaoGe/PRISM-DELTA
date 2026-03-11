"""PRISM model wrappers for inference-time attention steering."""
from .projection_builder_base import ProjectionBuilderBase
from .prism_llm import PrismLLM
from .adaptive_prism_llm import AdaptivePrismLLM

# Backward-compatible aliases.
SEKALLM = PrismLLM
AdaptiveSEKALLM = AdaptivePrismLLM
