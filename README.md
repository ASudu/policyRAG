# PolicyRAG Governance Evaluator

Local-first RAG application and evaluation framework for enterprise policy answers.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

Install the Ollama models used by the default configuration:

```powershell
ollama pull qwen3:8b
ollama pull gemma3:12b
ollama pull nomic-embed-text
```

Run the UI:

```powershell
uv run streamlit run app/streamlit_app.py
```

Run tests:

```powershell
uv run pytest
```

See `project.md` for the product requirements and implementation sequence.
