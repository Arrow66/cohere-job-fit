# Cohere Job Fit

A  retrieval pipeline on Cohere's actual product stack:

1. **Embed v4** — retrieve resume evidence for each intern requirement
2. **Rerank** — score those snippets against the posting
3. **Command** — write a grounded fit memo


## Setup

1. Create a trial key at [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys)
2. Copy `.env.example` to `.env` and set `CO_API_KEY`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python job_fit.py
```

## What it calls

| Step | Model | Why |
|---|---|---|
| Retrieve | `embed-v4.0` | `search_document` vs `search_query` input types |
| Rank | `rerank-v3.5` | Relevance over raw cosine |
| Write | `command-a-03-2025` | Memo grounded in reranked evidence |

Override models with `COHERE_CHAT_MODEL`, `COHERE_EMBED_MODEL`, `COHERE_RERANK_MODEL`.

## Files

- `data/resume_chunks.txt` — experience as retrieval units
- `data/cohere_intern.txt` — posting broken into requirements
- `job_fit.py` — the pipeline
- `examples/last_run.json` — last scores + memo (created after a run)

## Author

[Arjun Biju](https://github.com/Arrow66)
