"""
Configuration Management Module
================================
RAG 2025 Domain Expert - Production-Grade Retrieval System

AUTHOR: Shawn Slattery
GitHub: https://github.com/shawnslat
LinkedIn: https://www.linkedin.com/in/shawn-slattery-843654201/

Built: November 2025
Stack: LlamaIndex + Pinecone + Groq + Nomic + Firecrawl

PURPOSE:
--------
Centralized configuration management using a frozen dataclass pattern to ensure
immutability and type safety. All environment variables are loaded once at import
time and cached for the application lifecycle.

DESIGN PATTERN:
--------------
Singleton-like configuration object using frozen dataclass
- @dataclass(frozen=True): Prevents accidental mutation after initialization
- All fields have sensible defaults for development/testing
- Type hints provide IDE autocomplete and static type checking

USAGE:
------
from config import settings
print(settings.groq_api_key)  # Access configuration values
# settings.groq_api_key = "new"  # ❌ Raises FrozenInstanceError

CALLED BY:
----------
- app.py: UI configuration and sidebar display
- query_engine.py: API keys, model names, retrieval parameters
- ingest.py: Crawl settings, embedding configuration
- utils.py: Logging configuration
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file at module import time
# This runs ONCE when any module imports config.py
# Order of precedence: .env file < system environment variables
load_dotenv()


@dataclass(frozen=True)  # frozen=True makes the class immutable after __init__
class Settings:
    """
    Immutable configuration container for RAG system.
    
    WHY FROZEN?
    -----------
    Prevents bugs like:
        settings.llm_model = "wrong-model"  # Typo that could break production
    
    Instead raises: dataclasses.FrozenInstanceError
    
    FIELD CATEGORIES:
    -----------------
    1. API Keys (required for core functionality)
    2. Domain/Index Configuration (what to crawl, where to store)
    3. Model Configuration (which AI models to use)
    4. RAG Tuning Parameters (chunk sizes, thresholds)
    5. Performance Tuning (batch sizes, timeouts)
    6. Feature Flags (enable/disable optional features)
    7. File System Paths (where to store data)
    """
    
    # ============================================================================
    # API KEYS - Required for external services
    # ============================================================================
    
    groq_api_key: str = os.getenv("GROQ_API_KEY", "").strip()
    # USAGE: query_engine.py line 37 - Initialize Groq LLM client
    # FORMAT: "gsk_..." (starts with gsk_ prefix)
    # WHERE TO GET: https://console.groq.com/keys
    # WHY .strip()?: Remove accidental whitespace from .env file
    
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "").strip()
    # USAGE: query_engine.py line 39, ingest.py line 181
    # FORMAT: "pcsk_..." for serverless, "xxxx-xxxx-xxxx" for pod-based
    # WHERE TO GET: https://app.pinecone.io -> API Keys
    # CRITICAL: Without this, vector storage won't work

    nomic_api_key: str = os.getenv("NOMIC_API_KEY", "").strip()
    # USAGE: ingest/query embedding calls when using Nomic embeddings
    # NOTE: On Streamlit Cloud, set via secrets; no interactive `nomic login`.
    
    firecrawl_api_key: str = os.getenv("FIRECRAWL_API_KEY", "").strip()
    # USAGE: ingest.py line 180 - Initialize Firecrawl crawler
    # FORMAT: "fc-..."
    # WHERE TO GET: https://firecrawl.dev/app/api-keys
    # FREE TIER: 500 crawl credits
    
    # ============================================================================
    # DOMAIN & INDEX CONFIGURATION
    # ============================================================================
    
    domain_url: str = os.getenv("DOMAIN_URL", "https://arxiv.org/list/cs/recent")
    # USAGE: ingest.py line 104 - Starting point for web crawl
    # DEFAULT: arXiv computer science recent papers (2000+ papers)
    # ALTERNATIVES: 
    #   - Company docs: "https://docs.yourcompany.com"
    #   - Blog: "https://yourblog.com/articles"
    #   - Knowledge base: "https://help.yourproduct.com"
    # NOTE: Firecrawl will follow links from this starting page
    
    index_name: str = os.getenv("INDEX_NAME", "rag-2025-nov18")
    # USAGE: 
    #   - ingest.py line 82 - Create/connect to Pinecone index
    #   - query_engine.py line 40 - Query from this index
    # NAMING: Use descriptive names like "product-docs-prod", "blog-staging"
    # IMPORTANT: Changing this creates a NEW index (old data remains in old index)
    
    # ============================================================================
    # MODEL CONFIGURATION - Which AI models to use
    # ============================================================================
    
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v1.5")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "768"))
    # USAGE: 
    #   - ingest.py line 232 - Embed documents during ingestion
    #   - query_engine.py line 35 - Embed queries for retrieval
    # MUST MATCH: Ingestion and query MUST use same model
    # DIMENSION: Set EMBEDDING_DIM to match the model (e.g., 768 for nomic or gte-base)
    # ALTERNATIVES:
    #   - "thenlper/gte-base" (HuggingFace, 768-dim)
    #   - "all-MiniLM-L6-v2" (HuggingFace, 384-dim)
    # COST: Nomic free tier may exhaust; HuggingFace models run locally.
    
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-70b-instruct")
    # USAGE:
    #   - query_engine.py line 37 - Answer generation
    #   - ingest.py line 192 - Metadata enrichment (if enabled)
    #   - utils.py line 27 - HyDE hypothesis generation
    # CURRENT GROQ MODELS (Nov 2025):
    #   - "llama-3.3-70b-versatile" (RECOMMENDED, newest)
    #   - "llama-3.1-70b-versatile" (stable alternative)
    #   - "mixtral-8x7b-32768" (long context, 32K tokens)
    # NOTE: Model names change; check https://console.groq.com/docs/models
    
    reranker_model: str = os.getenv("RERANKER_MODEL", "mixedbread-ai/mxbai-rerank-base-v1")
    # USAGE: query_engine.py line 58 - Rerank retrieved chunks
    # PURPOSE: Improve relevance by cross-encoding query + chunk pairs
    # WHY CONFIGURABLE: Different models trade speed vs accuracy
    # OPTIONS:
    #   - "mxbai-rerank-xsmall-v1" (fastest, lower quality)
    #   - "mxbai-rerank-base-v1" (balanced, RECOMMENDED)
    #   - "mxbai-rerank-large-v1" (slowest, highest quality)
    # OPTIONAL: System works without reranker (see query_engine.py line 54-59)
    
    # ============================================================================
    # RAG TUNING PARAMETERS - Core retrieval settings
    # ============================================================================
    
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "768"))
    # USAGE: ingest.py line 237 - Split documents into chunks
    # WHY 768?: 
    #   - Matches embedding model context (Nomic processes ~512 tokens efficiently)
    #   - ~600-700 words of text
    #   - Balances context vs precision
    # TOO SMALL (128): Loses context, fragments sentences
    # TOO LARGE (4096): Dilutes relevance, mixes topics
    # TUNING: Experiment with 512, 768, 1024 for your domain
    
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    # USAGE: ingest.py line 237 - Overlapping tokens between chunks
    # PURPOSE: Prevent information loss at chunk boundaries
    # EXAMPLE:
    #   Chunk 1: "...the transformer architecture uses self-attention"
    #   Chunk 2: "self-attention mechanisms to process sequences..."
    #   (100 tokens overlap ensures "self-attention" context preserved)
    # RULE OF THUMB: 10-20% of chunk_size (100/768 = 13%)
    
    similarity_cutoff: float = float(os.getenv("SIMILARITY_CUTOFF", "0.77"))
    # USAGE: query_engine.py line 51 - Filter low-relevance chunks
    # RANGE: 0.0 (unrelated) to 1.0 (identical)
    # INTERPRETATION:
    #   - 0.9+: Very similar, almost exact match
    #   - 0.77: Moderately similar (current setting)
    #   - 0.6-0.7: Weakly similar, may include noise
    # TUNING: 
    #   - Increase (0.8-0.85) if getting too many irrelevant results
    #   - Decrease (0.7-0.75) if missing relevant results
    # DOMAIN DEPENDENT: Technical jargon needs lower threshold
    
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
    # USAGE: query_engine.py line 87 - Trigger web fallback
    # PURPOSE: Detect when internal knowledge is insufficient
    # LOGIC: if avg_score < 0.80 → search web for supplemental info
    # CALIBRATION PROCESS:
    #   1. Run 50-100 test queries
    #   2. Manually label quality (good/bad)
    #   3. Plot precision/recall at different thresholds
    #   4. Choose based on false positive tolerance
    # CONSERVATIVE (0.85): Fewer false positives, more web searches
    # AGGRESSIVE (0.75): Fewer web searches, more false positives
    
    # ============================================================================
    # PERFORMANCE TUNING - Batch processing and rate limits
    # ============================================================================
    
    embed_batch_sizes: tuple[int, ...] = tuple(
        int(item)
        for item in os.getenv("EMBED_BATCH_SIZES", "64,32,16,8").split(",")
        if item.strip()
    )
    # USAGE: ingest.py line 231-244 - Exponential backoff on rate limits
    # ALGORITHM:
    #   1. Try batch_size=64 (fastest)
    #   2. If rate limited (429 error) → retry with batch_size=32
    #   3. Continue: 32 → 16 → 8 until success or all fail
    # WHY DESCENDING?: Maximize throughput while handling rate limits gracefully
    # NOMIC LIMITS: ~200 requests/min, ~100 embeddings/request
    # CUSTOMIZATION: "128,64,32,16" for more aggressive, "32,16,8" for conservative
    
    # ============================================================================
    # OPTIONAL FEATURES & API KEYS
    # ============================================================================
    
    tavily_api_key: str | None = os.getenv("TAVILY_API_KEY", "").strip() or None
    # USAGE: utils.py line 77, query_engine.py line 94
    # PURPOSE: Live web search when confidence < threshold
    # TYPE: str | None (None if not provided)
    # WHY OPTIONAL?: System works without it (just no web fallback)
    # WHERE TO GET: https://tavily.com
    # FREE TIER: 1000 searches/month
    # ALTERNATIVE: Could use Exa, Brave Search API, or Google Custom Search
    
    enable_metadata_enrichment: bool = os.getenv("ENABLE_METADATA_ENRICHMENT", "false").lower() == "true"
    # USAGE: ingest.py line 191-207
    # PURPOSE: Use LLM to generate better titles + question metadata
    # COST IMPACT: Adds ~1 LLM call per document (can be expensive)
    # DEFAULT: false (uses heuristic titles to save API calls)
    # WHEN TO ENABLE:
    #   - Documents have poor/missing titles
    #   - Need better semantic search
    #   - Budget allows extra LLM calls
    # IMPLEMENTATION: ingest.py _enrich_metadata_with_llm()
    
    # ============================================================================
    # SYSTEM CONFIGURATION
    # ============================================================================
    
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # USAGE: All modules line ~18 - Configure logging verbosity
    # VALID VALUES:
    #   - "DEBUG": Everything (verbose, dev only)
    #   - "INFO": Normal operations (recommended)
    #   - "WARNING": Only problems
    #   - "ERROR": Only errors
    # CHANGES TO: logging.basicConfig(level=logging.INFO)
    # DEV TIP: Use "DEBUG" when troubleshooting, "INFO" for production
    
    # ============================================================================
    # FILE SYSTEM PATHS - Where to store data
    # ============================================================================
    
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", "storage"))
    # USAGE:
    #   - ingest.py line 263 - Persist indexes after ingestion
    #   - query_engine.py line 46 - Load indexes at startup
    # CONTENTS:
    #   - docstore.json: Document metadata and text
    #   - index_store.json: Index metadata
    #   - graph_store.json: Node relationships
    # SIZE: Can be large (multi-GB for big corpora)
    # GITIGNORE: Add to .gitignore (don't commit to git)
    # CLEANUP: Delete to force re-ingestion
    
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    # USAGE: ingest.py line 98 - Cache crawled pages
    # CONTENTS:
    #   - crawled_pages.jsonl: List of crawled URLs + titles
    # PURPOSE: 
    #   - Debugging: See what was crawled
    #   - Resume: Avoid re-crawling if ingestion fails
    #   - Auditing: Track data sources
    # SIZE: Small (just metadata, ~1KB per page)


# ============================================================================
# MODULE-LEVEL SINGLETON
# ============================================================================

# Create single instance at module import time
# All imports get the SAME instance (singleton pattern)
# USAGE: from config import settings (lowercase)
settings = Settings()

# WHY SINGLETON?
# --------------
# 1. Single source of truth for configuration
# 2. Loaded once, cached for application lifetime
# 3. No risk of stale/inconsistent config across modules
# 4. Easy to mock for testing: just patch `config.settings`

# TESTING EXAMPLE:
# ----------------
# from unittest.mock import patch
# with patch('config.settings.groq_api_key', 'test-key'):
#     # Test code that uses settings
#     pass
