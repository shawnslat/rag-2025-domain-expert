#!/usr/bin/env python
"""
Data Ingestion Pipeline Module
===============================
RAG 2025 Domain Expert - Web Crawling & Vector Indexing

AUTHOR: Shawn Slattery
GitHub: https://github.com/shawnslat
LinkedIn: https://www.linkedin.com/in/shawn-slattery-843654201/

Built: November 2025
Pipeline: Firecrawl → Chunk → Embed (Nomic) → Pinecone + Local Summary Index

NOTE: This is the ANNOTATED version with 1000+ lines of developer comments.
      For the working version, see ingest.py (uploaded file).
      This version explains every design decision and trade-off.

PURPOSE:
--------
Complete data ingestion pipeline that transforms raw website content into 
searchable vector embeddings. This is a ONE-TIME operation (or periodic refresh)
that prepares the knowledge base for the query engine.

PIPELINE STAGES:
----------------
1. Web Crawling (Firecrawl)
   - Crawl website starting from domain_url
   - Convert HTML → clean Markdown
   - Extract metadata (title, URL)
   - Cache to JSONL for debugging

2. Metadata Enrichment (Optional)
   - Use LLM to generate better titles
   - Extract 3 questions each document answers
   - Fallback to heuristics if LLM fails

3. Text Chunking
   - Split documents into 768-token chunks
   - 100-token overlap to preserve context
   - Maintain metadata across chunks

4. Embedding Generation
   - Convert text → 768-dim vectors (Nomic)
   - Batch processing with rate limit handling
   - Exponential backoff: 64 → 32 → 16 → 8

5. Index Creation
   - Upload vectors to Pinecone (cloud)
   - Create SummaryIndex locally (disk)
   - Persist for query engine

USAGE:
------
# Basic (ingest 50 pages, 10-minute timeout)
python ingest.py --total-pages 50 --crawl-timeout 600

# Full arXiv corpus (2000+ papers, 30-minute timeout)
python ingest.py --total-pages 500 --crawl-timeout 1800

# All available pages (no limit)
python ingest.py

CALLED BY:
----------
- Manual execution (command line)
- CI/CD pipelines (nightly data refresh)
- Initial setup scripts

PERFORMANCE:
------------
- 50 pages: ~2-5 minutes
- 500 pages: ~10-20 minutes
- 2000 pages: ~30-60 minutes

Bottlenecks:
- Firecrawl crawling: ~1-2s per page
- Nomic embedding: ~50-100ms per chunk
- Pinecone upsert: ~10-20ms per batch

COST:
-----
Per 500 pages (~5000 chunks):
- Firecrawl: 500 credits (~$0.50 if paid tier)
- Nomic: Free (up to rate limits)
- Pinecone: Free (serverless tier)
- Total: ~$0.50-1.00

DEPENDENCIES:
-------------
- firecrawl: Web crawling API
- llama_index: RAG framework
- pinecone: Vector database
- nomic: Embedding model
- groq: Optional LLM for metadata enrichment
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import argparse
import logging
import time
from json import JSONDecodeError
from typing import Tuple

import typer
from firecrawl import Firecrawl
from firecrawl.v2.types import PaginationConfig
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, SummaryIndex
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from config import settings

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Detect if an exception is a rate limit error.
    
    PURPOSE:
    --------
    Different APIs return rate limits in different ways:
    - HTTP 429 status code
    - "rate limit" in error message
    - "ratelimit" or "RateLimit" variations
    
    This function normalizes detection across all APIs.
    
    PARAMETERS:
    -----------
    exc : Exception
        Any exception that might be a rate limit error
        
    RETURNS:
    --------
    bool : True if rate limit detected, False otherwise
    
    USAGE:
    ------
    try:
        embed_documents()
    except Exception as e:
        if _is_rate_limit_error(e):
            # Back off and retry
            time.sleep(2)
        else:
            # Different error, re-raise
            raise
    
    CALLED BY:
    ----------
    - run() line 263: Exponential backoff on embedding
    """
    # Convert exception to lowercase string for case-insensitive matching
    message = str(exc).lower()
    
    # Check for common rate limit indicators in message
    # "429" → HTTP status code
    # "rate limit" → Explicit message
    # "ratelimit" → Alternate spelling
    if "429" in message or "rate limit" in message or "ratelimit" in message:
        return True
    
    # Check HTTP status code attribute (if present)
    # Some libraries attach .status_code to exceptions
    status_code = getattr(exc, "status_code", None)
    return status_code == 429


def _trim_text_for_llm(text: str, max_chars: int = 4000) -> str:
    """
    Truncate long text for LLM processing to avoid token limits.
    
    PROBLEM:
    --------
    Documents can be 10K-100K+ words, but metadata enrichment only needs
    a sample to generate a title. Sending full text wastes tokens and time.
    
    SOLUTION:
    ---------
    Take first 2000 chars + last 2000 chars (4000 total)
    This captures intro + conclusion, which usually contains key info.
    
    PARAMETERS:
    -----------
    text : str
        Full document text (could be very long)
    max_chars : int
        Maximum characters to return (default 4000)
        
    RETURNS:
    --------
    str : Trimmed text with "...[truncated]..." in middle if needed
    
    EXAMPLE:
    --------
    text = "Introduction to RAG..." + 100K words + "...Conclusion"
    trimmed = _trim_text_for_llm(text)
    # Returns: "Introduction to RAG...\n\n...[truncated]...\n\n...Conclusion"
    
    WHY 4000 CHARS?:
    ----------------
    - ~1000 tokens (4 chars per token rough average)
    - Leaves room for prompt + response
    - Fast to process (<1s LLM call)
    
    CALLED BY:
    ----------
    - _enrich_metadata_with_llm() line 82
    """
    # If text is already short enough, return as-is
    if len(text) <= max_chars:
        return text
    
    # Take first half from beginning
    head = text[: max_chars // 2]  # First 2000 chars
    
    # Take second half from end (using negative indexing)
    tail = text[-max_chars // 2 :]  # Last 2000 chars
    
    # Join with truncation marker
    return f"{head}\n\n...[truncated]...\n\n{tail}"


def _enrich_metadata_with_llm(doc: Document, llm: Groq) -> Tuple[str, list[str]]:
    """
    Use LLM to generate better document metadata for retrieval.
    
    PURPOSE:
    --------
    Crawled documents often have poor titles:
    - "Untitled" or "Document 1"
    - Generic page titles like "Home - Company Name"
    - Missing titles entirely
    
    LLM can read the content and generate:
    1. Descriptive title (e.g., "Introduction to Neural Networks")
    2. Questions the document answers (improves retrieval)
    
    COST:
    -----
    1 LLM call per document (~$0.0001 per doc with Groq)
    For 500 docs: ~$0.05
    
    CONTROLLED BY:
    --------------
    ENABLE_METADATA_ENRICHMENT=true in .env
    Default: false (uses heuristic fallback)
    
    PARAMETERS:
    -----------
    doc : Document
        LlamaIndex Document object with .text and .metadata
    llm : Groq
        Groq LLM client for completion
        
    RETURNS:
    --------
    Tuple[str, list[str]]
        (title, [question1, question2, question3])
        
    EXAMPLE OUTPUT:
    ---------------
    title = "Transformer Architecture Explained"
    questions = [
        "What is the transformer architecture?",
        "How does self-attention work?",
        "Why are transformers better than RNNs?"
    ]
    
    FALLBACK:
    ---------
    If LLM fails (network error, parsing error):
    - Use original title or "Untitled"
    - Generate 3 generic questions
    
    CALLED BY:
    ----------
    - run() line 223: For each document if enrichment enabled
    """
    # Construct prompt for structured JSON output
    # CRITICAL: "strict JSON" → LLM must return valid JSON
    prompt = (
        "You are enriching crawled documents for retrieval. "
        "Given the markdown below, extract a concise title (max 12 words) "
        "and three concrete questions the document answers. "
        "Return strict JSON with keys 'title' and 'questions_answered'.\n\n"
        f"Content:\n{_trim_text_for_llm(doc.text)}\n\nJSON:"
    )
    # PROMPT ENGINEERING:
    # - "concise title (max 12 words)" → Prevents rambling titles
    # - "concrete questions" → Encourages specific, searchable questions
    # - "strict JSON" → Reduces parsing errors
    
    try:
        # Call LLM (synchronous, blocks ~1-2 seconds)
        completion = llm.complete(prompt)
        # RESPONSE EXAMPLE:
        # {
        #   "title": "Introduction to Transformer Networks",
        #   "questions_answered": [
        #     "What are transformer networks?",
        #     "How does attention mechanism work?",
        #     "Why use transformers over RNNs?"
        #   ]
        # }
        
        # Parse JSON response
        payload = json.loads(completion.text)
        # RISK: LLM might return invalid JSON or wrap in markdown
        # HANDLED: Try/except catches JSONDecodeError
        
        # Extract title with fallback
        title = payload.get("title") or doc.metadata.get("title") or "Untitled"
        # LOGIC: LLM title > original title > "Untitled"
        
        # Extract questions with validation
        questions = payload.get("questions_answered") or []
        # EDGE CASE: LLM might return string instead of list
        if isinstance(questions, str):
            questions = [questions]  # Wrap single string in list
        
        # Clean and filter questions
        return title, [str(q).strip() for q in questions if q]
        # FILTERING:
        # - str(q): Convert to string (in case of non-string)
        # - .strip(): Remove whitespace
        # - if q: Remove None/empty strings
        
    except (JSONDecodeError, Exception) as exc:  # noqa: BLE001
        # Catch both JSON parsing errors and any other errors
        # PHILOSOPHY: Graceful degradation > crashing ingestion
        
        logger.warning("Metadata enrichment failed, falling back to heuristics: %s", exc)
        # LOG: Track failures for debugging, but don't crash
        
        # Fallback to heuristic title generation
        title = doc.metadata.get("title") or "Untitled"
        
        # Generate generic but useful questions
        return title, [
            f"What is {title} about?",
            f"How does {title} approach the problem?",
            f"Why is {title} impactful for practitioners?",
        ]
        # HEURISTIC QUESTIONS:
        # - Still better than nothing
        # - Use title to make questions specific
        # - Cover common query patterns (what, how, why)


def _build_embed_model(batch_size: int):
    """
    Build an embedding model instance based on configuration.
    - Nomic models: use NomicEmbedding with batch size
    - Other models: use HuggingFaceEmbedding (local/hosted)
    """
    if settings.embedding_model.startswith("nomic"):
        return NomicEmbedding(
            model_name=settings.embedding_model,
            embed_batch_size=batch_size,
            api_key=settings.nomic_api_key,
        )
    return HuggingFaceEmbedding(
        model_name=settings.embedding_model,
        embed_batch_size=batch_size,
    )


# ============================================================================
# PINECONE INDEX MANAGEMENT
# ============================================================================

def ensure_index(pc: Pinecone) -> None:
    """
    Create Pinecone index if it doesn't exist.
    
    PURPOSE:
    --------
    Idempotent index creation (safe to call multiple times)
    - If index exists: Do nothing
    - If index missing: Create with correct settings
    
    INDEX CONFIGURATION:
    --------------------
    - Dimension: settings.embedding_dim (matches embedding model)
    - Metric: cosine (best for semantic similarity)
    - Spec: Serverless AWS us-east-1 (free tier available)
    
    PARAMETERS:
    -----------
    pc : Pinecone
        Pinecone client instance
        
    RETURNS:
    --------
    None (creates index as side effect)
    
    CALLED BY:
    ----------
    - run() line 195: Before crawling/indexing
    
    SERVERLESS VS POD:
    ------------------
    Serverless (chosen):
    - Pay per request (free tier: 1M vectors)
    - Auto-scaling
    - No infrastructure management
    
    Pod (not used):
    - Fixed cost (~$70/month minimum)
    - Better for high throughput
    - Manual scaling
    
    REGION CHOICE:
    --------------
    us-east-1: Lowest latency for US East Coast
    ALTERNATIVES:
    - eu-west-1: Europe
    - asia-southeast1: Asia Pacific
    """
    # Check if index already exists
    # list_indexes() returns list of IndexModel objects
    existing_names = [idx.name for idx in pc.list_indexes()]
    
    if settings.index_name not in existing_names:
        # Index doesn't exist, create it
        pc.create_index(
            name=settings.index_name,  # From .env: INDEX_NAME
            dimension=settings.embedding_dim,  # MUST match embedding model dimension
            metric="cosine",  # Similarity metric
            # OPTIONS:
            # - "cosine": Angle between vectors (best for semantic)
            # - "euclidean": Distance between vectors
            # - "dotproduct": Dot product (requires normalized vectors)
            # CHOSEN: Cosine for semantic similarity (standard for RAG)
            
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
            # Serverless configuration
            # CLOUD: aws (also supports gcp)
            # REGION: us-east-1 (lowest latency for US East Coast)
        )
        typer.secho(f"Created Pinecone index '{settings.index_name}'", fg="green")
        # USER FEEDBACK: Visual confirmation of index creation
    # If index exists, do nothing (idempotent)


def load_local_docs(data_dir: Path) -> list[Document]:
    """Load local .md/.txt/.json files from data_dir into Documents."""
    docs: list[Document] = []
    if not data_dir.exists():
        return docs
    for path in data_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt", ".json"}:
            continue
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                text = payload.get("text") or payload.get("content") or json.dumps(payload)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping %s (read error): %s", path, exc)
            continue
        if not text.strip():
            continue
        docs.append(
            Document(
                text=text[:1_000_000],
                metadata={
                    "source_url": str(path.resolve()),
                    "title": path.stem or "Untitled",
                },
            )
        )
    return docs


# ============================================================================
# WEB CRAWLING
# ============================================================================

def crawl(
    firecrawl_client: Firecrawl,
    total_pages: Optional[int],
    page_size: int,
    crawl_timeout: int,
    poll_interval: int,
) -> list[Document]:
    """
    Crawl website and convert to LlamaIndex Document objects.
    
    PROCESS:
    --------
    1. Start Firecrawl job (async, returns immediately)
    2. Poll status every poll_interval seconds
    3. On completion, fetch all pages
    4. Convert HTML → Markdown
    5. Create Document objects with metadata
    6. Cache to JSONL for debugging
    
    PARAMETERS:
    -----------
    firecrawl_client : Firecrawl
        Authenticated Firecrawl API client
    total_pages : Optional[int]
        Max pages to crawl (None = unlimited)
    page_size : int
        Batch size for progress reporting (cosmetic)
    crawl_timeout : int
        Max seconds to wait for crawl completion
    poll_interval : int
        Seconds between status checks
        
    RETURNS:
    --------
    list[Document] : LlamaIndex Document objects
        Each with .text (markdown) and .metadata (url, title)
        
    EXAMPLE:
    --------
    docs = crawl(client, total_pages=50, page_size=10, ...)
    # Returns: [Document(text="# Title\n\nContent...", metadata={...}), ...]
    
    FIRECRAWL PROCESS:
    ------------------
    1. start_crawl() → Returns job ID immediately
    2. Firecrawl crawls in background (1-30 minutes)
    3. Poll get_crawl_status() until status="completed"
    4. Fetch results with auto_paginate=True
    
    TIMEOUT HANDLING:
    -----------------
    - Short timeout (300s): For testing/debugging
    - Long timeout (1800s): For large sites
    - On timeout: Raise TimeoutError (not partial results)
    
    CACHING:
    --------
    Writes to data/crawled_pages.jsonl:
    {"url": "https://...", "title": "Page Title"}
    {"url": "https://...", "title": "Another Page"}
    
    PURPOSE: Debugging + manual inspection of crawled URLs
    
    CALLED BY:
    ----------
    - run() line 198: After Pinecone setup, before embedding
    """
    # Setup cache file for debugging
    cache_file = settings.data_dir / "crawled_pages.jsonl"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # CREATE: data/ directory if doesn't exist
    # exist_ok=True: Don't error if already exists

    docs: list[Document] = []  # Accumulator for results
    start_url = settings.domain_url.rstrip("/")  # Remove trailing slash
    # WHY rstrip("/")?:
    # "https://example.com/" → "https://example.com"
    # Firecrawl prefers URLs without trailing slash

    # ========================================================================
    # STAGE 1: START CRAWL JOB
    # ========================================================================
    
    logger.info("Starting crawl of %s", start_url)
    typer.secho(f"Starting crawl of {start_url}", fg="blue")
    # DUAL OUTPUT: Log (for automation) + terminal (for humans)
    
    job = firecrawl_client.start_crawl(
        start_url,  # Starting point
        limit=total_pages,  # Max pages (None = unlimited)
        scrape_options={
            "only_main_content": True,  # Skip nav, footer, ads
            "formats": ["markdown"],  # Return clean Markdown
        },
        exclude_paths=[r".*pdf.*"],  # Skip PDFs (can't parse well)
        # REGEX: r".*pdf.*" matches any URL containing "pdf"
        # EXAMPLES:
        # - https://example.com/paper.pdf → SKIP
        # - https://example.com/docs → CRAWL
    )
    # RETURNS: Job object with .id
    # CRAWL: Happens asynchronously on Firecrawl servers

    # ========================================================================
    # STAGE 2: POLL FOR COMPLETION
    # ========================================================================
    
    typer.secho(
        f"Firecrawl job {job.id} launched – polling status every {poll_interval}s", 
        fg="blue"
    )
    logger.info("Firecrawl job %s launched", job.id)
    
    start_time = time.time()  # For timeout tracking
    pages: list = []  # Will hold final results
    
    # Polling configuration (don't auto-paginate yet)
    polling_config = PaginationConfig(auto_paginate=False)
    # WHY?: We just want status, not full results (saves bandwidth)
    
    while True:
        # Manual polling loop
        # ALTERNATIVE: Could use webhooks, but polling is simpler
        
        # Fetch current status
        status = firecrawl_client.get_crawl_status(
            job.id, 
            pagination_config=polling_config
        )
        
        # Display progress to user
        typer.secho(
            f"[Firecrawl] status={status.status} "
            f"completed={status.completed}/{status.total} "
            f"credits={status.credits_used}",
            fg="magenta",
        )
        logger.info(
            "Firecrawl status=%s completed=%s/%s credits=%s",
            status.status,
            status.completed,
            status.total,
            getattr(status, "credits_used", "n/a"),
        )
        # STATUS VALUES:
        # - "scraping": Still crawling
        # - "completed": Done successfully
        # - "failed": Error occurred
        # - "cancelled": User cancelled
        
        # Check for completion
        if status.status == "completed":
            # SUCCESS: Crawl finished
            
            # Now fetch full results with pagination
            final_config = PaginationConfig(
                auto_paginate=True,  # Get all results automatically
                max_results=total_pages,  # Cap at our limit
            )
            final_status = firecrawl_client.get_crawl_status(
                job.id, 
                pagination_config=final_config
            )
            pages = final_status.data or []
            # RESULT: List of page objects with .markdown, .metadata
            break  # Exit polling loop
        
        # Check for failure
        if status.status in {"failed", "cancelled"}:
            # FAILURE: Something went wrong
            raise RuntimeError(
                f"Crawl {status.status}. "
                f"Check Firecrawl dashboard for job {job.id}."
            )
            # USER ACTION: Visit https://firecrawl.dev/dashboard
        
        # Check for timeout
        if crawl_timeout and (time.time() - start_time) > crawl_timeout:
            # TIMEOUT: Took too long
            raise TimeoutError(
                f"Crawl timed out after {crawl_timeout}s. "
                f"Use --crawl-timeout to increase or retry with smaller --total-pages."
            )
            # USER GUIDANCE: How to fix the problem
        
        # Wait before next poll
        time.sleep(poll_interval)
        # TYPICAL: 5 seconds (balance between responsiveness and API load)

    # ========================================================================
    # STAGE 3: PROCESS RESULTS
    # ========================================================================
    
    limit = total_pages or len(pages)  # Use total_pages if set, else all pages
    batches = math.ceil(limit / page_size) if page_size else 1
    # COSMETIC: For progress bar estimation
    
    typer.secho(
        f"Processing up to {limit} pages in ~{batches} batches of {page_size}", 
        fg="blue"
    )
    logger.info("Processing up to %s pages", limit)

    # Process each page with progress bar
    for page in tqdm(pages[:limit], desc="Processing pages", total=min(limit, len(pages))):
        # PROGRESS BAR: tqdm shows: Processing pages: 45%|████▌     | 23/50
        
        # Extract markdown content
        md = getattr(page, "markdown", None)
        # SAFE: getattr with None default (page might not have .markdown)
        
        # Skip pages with little content
        if not md or len(md) < 200:
            continue
        # THRESHOLD: 200 chars minimum (filters out empty/stub pages)
        
        # Extract metadata safely
        metadata = {}
        try:
            metadata = page.metadata_dict  # type: ignore[attr-defined]
            # Firecrawl v2 API format
        except AttributeError:
            # Fallback for older API or missing attribute
            md_raw = getattr(page, "metadata", {}) or {}
            if isinstance(md_raw, dict):
                metadata = md_raw
        
        # Extract URL and title with fallbacks
        source_url = (
            metadata.get("url") or 
            metadata.get("sourceUrl") or  # Alternate key name
            start_url  # Last resort fallback
        )
        title = metadata.get("title") or "Untitled"
        
        # Create LlamaIndex Document object
        doc = Document(
            text=md[:1_000_000],  # Cap at 1M chars (prevents memory issues)
            metadata={
                "source_url": source_url,
                "title": title,
            }
        )
        docs.append(doc)
        
        # Cache to JSONL for debugging
        with cache_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"url": source_url, "title": title}) + "\n")
        # FORMAT: One JSON object per line (easy to parse/grep)
        # USAGE: grep "arxiv" data/crawled_pages.jsonl
        
        # Check if we've hit the limit
        if total_pages and len(docs) >= total_pages:
            break
        # EARLY EXIT: Stop processing if we hit target count

    typer.secho(f"Crawled & cached {len(docs)} documents", fg="green")
    logger.info("Crawled and cached %s documents", len(docs))
    
    return docs
    # RETURNS: List of Document objects ready for embedding


# ============================================================================
# MAIN INGESTION ORCHESTRATION
# ============================================================================

def run(
    total_pages: Optional[int], 
    page_size: int, 
    crawl_timeout: int, 
    poll_interval: int,
    use_local: bool,
) -> None:
    """
    Main orchestration function for complete ingestion pipeline.
    
    FULL PIPELINE:
    --------------
    1. Validate API keys
    2. Initialize clients (Firecrawl, Pinecone)
    3. Ensure Pinecone index exists
    4. Crawl website → Documents
    5. Enrich metadata (optional, LLM-based)
    6. Chunk documents → Nodes
    7. Embed nodes → Vectors (with batch size backoff)
    8. Create vector index (Pinecone)
    9. Create summary index (local storage)
    10. Persist indexes to disk
    
    PARAMETERS:
    -----------
    total_pages : Optional[int]
        Max pages to crawl (None = unlimited)
    page_size : int
        Batch size for progress (cosmetic)
    crawl_timeout : int
        Max seconds for crawl
    poll_interval : int
        Seconds between status polls
        
    RETURNS:
    --------
    None (creates indexes as side effect)
    
    CALLED BY:
    ----------
    - main() line 302: CLI entry point
    
    ERROR HANDLING:
    ---------------
    - Missing API keys → Exit with error message
    - Crawl failure → RuntimeError with job ID
    - Crawl timeout → TimeoutError with guidance
    - Embedding rate limit → Exponential backoff
    - Embedding failure → Re-raise after retries
    """
    # ========================================================================
    # STAGE 1: VALIDATION
    # ========================================================================
    
    # Check for required API keys
    if not settings.pinecone_api_key:
        typer.secho("Missing PINECONE_API_KEY", fg="red")
        raise typer.Exit(1)
    if not use_local and not settings.firecrawl_api_key:
        typer.secho("Missing FIRECRAWL_API_KEY; pass --use-local to ingest local files instead.", fg="red")
        raise typer.Exit(1)
    # FAIL FAST: Better to error immediately than halfway through
    
    # ========================================================================
    # STAGE 2: CLIENT INITIALIZATION
    # ========================================================================
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=settings.pinecone_api_key)
    # CREATES: gRPC connection to Pinecone API
    
    # Create index if doesn't exist
    ensure_index(pc)
    # IDEMPOTENT: Safe to call multiple times

    # ========================================================================
    # STAGE 3: WEB CRAWLING OR LOCAL LOAD
    # ========================================================================
    if use_local:
        docs = load_local_docs(settings.data_dir)
        typer.secho(f"Loaded {len(docs)} local documents from {settings.data_dir}", fg="blue")
        logger.info("Loaded %s local documents from %s", len(docs), settings.data_dir)
    else:
        # Initialize Firecrawl client
        firecrawl_client = Firecrawl(api_key=settings.firecrawl_api_key)
        # CREATES: HTTP client with authentication
        
        docs = crawl(
            firecrawl_client,
            total_pages=total_pages,
            page_size=page_size,
            crawl_timeout=crawl_timeout,
            poll_interval=poll_interval,
        )
        # DURATION: 2-30 minutes depending on site size
        # OUTPUT: List of Document objects
    
    if not docs:
        typer.secho("No documents retrieved – check DOMAIN_URL or local data folder", fg="red")
        raise typer.Exit(1)
    # VALIDATION: Ensure we got something

    # ========================================================================
    # STAGE 4: METADATA ENRICHMENT (OPTIONAL)
    # ========================================================================
    
    llm = None
    if settings.enable_metadata_enrichment and settings.groq_api_key:
        # Both flag enabled AND API key present
        llm = Groq(
            model=settings.llm_model, 
            api_key=settings.groq_api_key, 
            temperature=0.1
        )
        logger.info("Metadata enrichment enabled using Groq model %s", settings.llm_model)
    elif settings.enable_metadata_enrichment and not settings.groq_api_key:
        # Flag enabled but API key missing
        logger.warning(
            "Metadata enrichment requested but GROQ_API_KEY is missing; "
            "falling back to heuristics."
        )
    else:
        # Flag disabled (default)
        logger.info("Metadata enrichment disabled via settings.")

    # Enrich each document
    for doc in docs:
        if llm:
            # LLM-based enrichment
            title, questions = _enrich_metadata_with_llm(doc, llm)
            # COST: ~$0.0001 per doc
        else:
            # Heuristic fallback
            title = doc.metadata.get("title") or "Untitled"
            questions = [
                f"What is {title} about?",
                f"How does {title} approach the problem?",
                f"Why is {title} impactful for practitioners?",
            ]

        # Update document metadata
        doc.metadata["document_title"] = title
        doc.metadata.setdefault("title", title)  # Don't overwrite if exists
        doc.metadata["questions_answered"] = questions
    # RESULT: All documents have title + 3 questions

    # ========================================================================
    # STAGE 5: EMBEDDING WITH EXPONENTIAL BACKOFF
    # ========================================================================
    
    nodes: list = []  # Will hold chunked + embedded nodes
    errors: list[Exception] = []  # Track failed attempts
    
    # Try each batch size in descending order
    for batch_size in settings.embed_batch_sizes:
        Settings.embed_model = NomicEmbedding(
            model_name=settings.embedding_model,
            api_key=settings.nomic_api_key,        # <-- This MUST be here
            embed_batch_size=batch_size,
        )
        # BATCH SIZE: How many chunks to embed in parallel
        # TRADE-OFF:
        # - Larger batch (64): Faster, but more likely to hit rate limits
        # - Smaller batch (8): Slower, but safer for rate limits
        
        # Create ingestion pipeline
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=settings.chunk_size,  # 768 tokens
                    chunk_overlap=settings.chunk_overlap,  # 100 tokens
                ),
                # CHUNKING: Split documents into 768-token chunks
                # OVERLAP: Preserve context across boundaries
                
                Settings.embed_model,
                # EMBEDDING: Convert text → 768-dim vectors
            ]
        )
        
        try:
            logger.info("Embedding with batch_size=%s", batch_size)
            
            # Run pipeline (chunking + embedding)
            nodes = pipeline.run(documents=docs, num_workers=1)
            # DURATION: ~1-5 minutes for 50 docs
            # num_workers=1: Single-threaded (safer for API limits)
            # OUTPUT: List of Node objects with embeddings
            
            break  # SUCCESS! Exit loop
            
        except Exception as exc:  # noqa: BLE001
            # Catch any exception
            errors.append(exc)
            
            if _is_rate_limit_error(exc):
                # RATE LIMIT: Too many requests
                logger.warning(
                    "Rate limit at batch_size=%s, backing off. Error: %s", 
                    batch_size, 
                    exc
                )
                time.sleep(2)  # Brief pause before retry
                continue  # Try next (smaller) batch size
            else:
                # Different error, can't recover
                raise
        # EXPONENTIAL BACKOFF:
        # Attempt 1: batch_size=64, fails → wait 2s
        # Attempt 2: batch_size=32, fails → wait 2s
        # Attempt 3: batch_size=16, fails → wait 2s
        # Attempt 4: batch_size=8, fails → give up

    # Check if all attempts failed
    if not nodes and errors:
        logger.error("Embedding failed after retries: %s", errors[-1])
        raise errors[-1]
    # FAILURE: Re-raise the last error

    typer.secho(f"Created {len(nodes)} nodes", fg="green")
    logger.info("Created %s nodes", len(nodes))
    # TYPICAL: 500 docs → 5000 nodes (10 chunks per doc average)

    # ========================================================================
    # STAGE 6: INDEX CREATION
    # ========================================================================
    
    # Setup vector store connection
    vector_store = PineconeVectorStore(
        pinecone_index=pc.Index(settings.index_name)
    )
    # CONNECTS: To the Pinecone index we created earlier
    
    # Ensure storage directory exists
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    # CREATE: storage/ directory for local indexes
    
    # Create storage context
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,  # Pinecone (cloud)
        docstore=SimpleDocumentStore(),  # In-memory initially
        index_store=SimpleIndexStore(),  # In-memory initially
        persist_dir=str(settings.storage_dir),  # Where to save locally
    )
    # DUAL STORAGE:
    # - Vectors → Pinecone (cloud, persistent)
    # - Metadata → Local disk (storage/, persistent)

    # Create vector index (uploads to Pinecone)
    vector_index = VectorStoreIndex(
        nodes, 
        storage_context=storage_context, 
        show_progress=True
    )
    # UPLOADS: All 5000 vectors to Pinecone
    # DURATION: ~30-60 seconds for 5000 vectors
    # PROGRESS: Shows upload progress bar
    
    # Create summary index (stays local)
    summary_index = SummaryIndex(
        nodes, 
        storage_context=storage_context
    )
    # PURPOSE: Document-level summaries for hierarchical retrieval
    # STORAGE: Local only (not uploaded to Pinecone)

    # ========================================================================
    # STAGE 7: PERSISTENCE
    # ========================================================================
    
    # Assign index IDs for loading later
    vector_index.set_index_id("vector")
    summary_index.set_index_id("summary")
    # IDs: Used by query_engine.py to load correct indexes
    
    # Persist to disk
    storage_context.persist(persist_dir=str(settings.storage_dir))
    # WRITES:
    # - storage/docstore.json (document metadata)
    # - storage/index_store.json (index configuration)
    # - storage/graph_store.json (node relationships)

    typer.secho(
        f"Success! Indexed {len(nodes)} nodes → "
        f"Pinecone '{settings.index_name}' + "
        f"summary cache at '{settings.storage_dir}'",
        fg="cyan"
    )
    # SUCCESS MESSAGE: Confirms completion


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Command-line interface entry point.
    
    USAGE:
    ------
    python ingest.py --total-pages 50 --crawl-timeout 600
    python ingest.py --help
    
    ARGUMENTS:
    ----------
    --total-pages : int (optional)
        Max pages to crawl
        Default: None (unlimited)
        
    --page-size : int
        Batch size for progress (cosmetic only)
        Default: 100
        
    --crawl-timeout : int
        Max seconds to wait for crawl
        Default: 900 (15 minutes)
        
    --poll-interval : int
        Seconds between status checks
        Default: 5
    """
    parser = argparse.ArgumentParser(
        description="One-click ingestion: crawl → chunk → embed → Pinecone",
    )
    parser.add_argument(
        "--total-pages",
        type=int,
        default=None,
        help="Limit the crawl to this many pages (default: all returned by Firecrawl).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="Informational batch size used for progress estimates.",
    )
    parser.add_argument(
        "--crawl-timeout",
        type=int,
        default=900,
        help="Seconds to wait for Firecrawl to finish before aborting (default: 15 minutes).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between Firecrawl status polls.",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Ingest local .md/.txt/.json files from DATA_DIR instead of crawling with Firecrawl.",
    )
    args = parser.parse_args()
    
    # Execute main pipeline
    run(
        total_pages=args.total_pages,
        page_size=args.page_size,
        crawl_timeout=args.crawl_timeout,
        poll_interval=args.poll_interval,
        use_local=args.use_local,
    )


if __name__ == "__main__":
    main()


# ============================================================================
# TROUBLESHOOTING GUIDE
# ============================================================================

"""
COMMON ISSUES:
--------------

1. "Missing FIRECRAWL_API_KEY"
   SOLUTION: Add to .env file
   
2. "Crawl timed out after 900s"
   SOLUTION: Increase --crawl-timeout or reduce --total-pages
   
3. "Rate limit at batch_size=64"
   SOLUTION: System auto-retries with smaller batches (wait ~30s)
   
4. "No documents retrieved"
   SOLUTION: Check DOMAIN_URL is accessible
   
5. "Embedding dimension mismatch"
   SOLUTION: Delete Pinecone index, re-run ingestion
   
6. "Out of memory"
   SOLUTION: Reduce --total-pages or increase system RAM

OPTIMIZATION TIPS:
------------------

1. Fast Ingestion:
   - Use higher embed_batch_sizes: "128,64,32"
   - Disable metadata enrichment: ENABLE_METADATA_ENRICHMENT=false
   
2. Better Quality:
   - Enable metadata enrichment: ENABLE_METADATA_ENRICHMENT=true
   - Increase chunk_size: CHUNK_SIZE=1024
   
3. Save Costs:
   - Crawl fewer pages: --total-pages 100
   - Disable metadata enrichment
   - Use free tiers only

MONITORING:
-----------

Watch logs for:
- "Rate limit at batch_size=X" → System handling backoff correctly
- "Embedding with batch_size=X" → Current batch size
- "Created X nodes" → Total chunks created
- "Success! Indexed X nodes" → Final confirmation
"""
