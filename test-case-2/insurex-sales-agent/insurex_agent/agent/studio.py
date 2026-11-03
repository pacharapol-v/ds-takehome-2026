"""Entry point for LangGraph Studio / `langgraph dev` (the platform supplies its own checkpointer)."""

from ..config import get_settings
from ..leads.gateway import make_gateway
from ..retrieval.hybrid import HybridRetriever
from .brain import make_brain
from .graph import SalesAgentGraph

_s = get_settings()
graph = SalesAgentGraph(make_brain(_s), HybridRetriever(settings=_s), make_gateway(_s), _s).build()
