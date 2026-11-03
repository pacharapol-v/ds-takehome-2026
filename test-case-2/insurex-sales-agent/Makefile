PY ?= python

install:            ## create venv deps
	$(PY) -m pip install -r requirements.txt

index:              ## parse PDFs + build Chroma/BM25 index (uses EMBEDDING_PROVIDER from .env)
	$(PY) -m insurex_agent.ingestion.build_index

chat:               ## interactive chat (debug shows the LangGraph node trace)
	$(PY) -m insurex_agent.cli chat --debug

demo:               ## scripted multi-session demo -> logs/demo_<llm>.log
	rm -rf data/state && $(PY) scripts/run_demo.py

eval:               ## retrieval ablation + end-to-end answer evaluation
	$(PY) scripts/eval_retrieval.py && $(PY) scripts/eval_agent.py

test:               ## offline test suite (no API key needed)
	$(PY) -m pytest -q

mcp-inspect:        ## open the MCP Inspector on the leads server (needs Node.js)
	npx @modelcontextprotocol/inspector $(PY) -m insurex_agent.leads.mcp_server

.PHONY: install index chat demo eval test mcp-inspect
