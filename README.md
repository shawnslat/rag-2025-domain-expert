# RAG 2025 Domain Expert

Production-ready RAG stack: crawl any domain with Firecrawl, chunk+embed with Nomic, store vectors in Pinecone Serverless, and answer with Groq Llama 3.3 via a Streamlit chatbot (HyDE + hierarchical retrieval, optional reranking).

## Accounts you need (free tiers)
| Service | Link | Notes |
| --- | --- | --- |
| Groq | https://console.groq.com | Create an API key (e.g., `rag-2025`). |
| Pinecone Serverless | https://app.pinecone.io | Create a serverless index (AWS us-east-1). |
| Firecrawl | https://firecrawl.dev | Create an API key (credits-based). |
| Tavily (optional fallback) | https://app.tavily.com | Enables live web search when internal confidence is low. |

## Quick start
```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env .env.local  # or edit .env directly
```

### `.env` essentials
```
# Domain + index
DOMAIN_URL=https://arxiv.org/list/cs/recent?show=2000, https://arxiv.org/list/physics/recent?show=1000
INDEX_NAME=rag-domain-expert-2025

# API keys
GROQ_API_KEY=...
PINECONE_API_KEY=...
FIRECRAWL_API_KEY=...
NOMIC_API_KEY=...                  # needed for embeddings
TAVILY_API_KEY=...                 # optional web fallback

# Models / params
EMBEDDING_MODEL=nomic-embed-text-v1.5
EMBEDDING_DIM=768                 # match your embedding model
NOMIC_API_KEY=...                 # required if using Nomic embeddings
LLM_MODEL=llama-3.3-70b-versatile   # update if Groq deprecates
RERANKER_MODEL=mixedbread-ai/mxbai-rerank-base-v1
CHUNK_SIZE=768
CHUNK_OVERLAP=100
SIMILARITY_CUTOFF=0.77
CONFIDENCE_THRESHOLD=0.80
EMBED_BATCH_SIZES=64,32,16,8
LOG_LEVEL=INFO
ENABLE_METADATA_ENRICHMENT=false    # set true to use Groq LLM for titles/Qs
```

## Ingest content
```bash
python ingest.py --total-pages 500 --page-size 100 --crawl-timeout 900 --poll-interval 5
```
- Crawls `DOMAIN_URL` via Firecrawl (markdown only, skips PDFs).
- Splits with SentenceSplitter → embeds with Nomic (adaptive batch backoff).
- Writes vectors to Pinecone and a local SummaryIndex (`storage/`) for hierarchical retrieval.
- Use `--total-pages` to cap pages; adjust timeout/polling for large crawls.
- Enable richer metadata extraction by setting `ENABLE_METADATA_ENRICHMENT=true` (uses Groq tokens).
- To ingest local files instead of crawling, drop `.md/.txt/.json` into `data/` and run `python ingest.py --use-local`.

## Run the chat UI
```bash
streamlit run app.py
```
Low-confidence answers will optionally call Tavily if `TAVILY_API_KEY` is set. Mixedbread reranker is optional; without it, similarity filtering still runs.

## Using the engine in code
```python
from query_engine import query_engine
bundle = query_engine("What trends are visible in the latest cs arXiv feed?")
text = "".join(bundle.response.response_gen)  # consume stream
print(text)
```

## Troubleshooting
- **Groq 429s or model errors**: use a current model name (e.g., `llama-3.3-70b-versatile`) and retry; 429s will back off automatically.
- **Mixedbread warning**: reranker not installed; either ignore or add a mixedbread rerank postprocessor in a newer LlamaIndex release.
- **Low-confidence blanks**: set `TAVILY_API_KEY` for web fallback or ingest more domains.
- **FileNotFound in storage**: the ingestion script now initializes fresh doc/index stores; rerun ingest.

## Ready for GitHub
- Secrets stay in `.env`; `.gitignore` excludes `.env`, `storage/`, `data/`, `venv/`, `__pycache__/`, and OS junk.
- To publish: `git init`, add remote, `git add .`, `git commit -m "Initial RAG stack"`, then push to GitHub. Streamlit Cloud can deploy directly from the repo.
