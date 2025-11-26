"""
Query Engine Module
===================
RAG 2025 Domain Expert - Core Retrieval Pipeline

AUTHOR: Shawn Slattery
GitHub: https://github.com/shawnslat
LinkedIn: https://www.linkedin.com/in/shawn-slattery-843654201/

Built: November 2025
Architecture: HyDE + Vector Retrieval + Neural Reranking + LLM Synthesis + Web Fallback

PURPOSE:
--------
Core retrieval and answer generation pipeline for RAG system.
Orchestrates the entire query flow:
1. Query Enhancement (HyDE)
2. Vector Retrieval (Pinecone)
3. Reranking (Mixedbread AI)
4. Answer Generation (Groq LLM)
5. Confidence Scoring
6. Web Fallback (Tavily)

ARCHITECTURE:
-------------
Simplified version (no recursive retrieval):
- Direct Pinecone vector search
- No dependency on local SummaryIndex
- More reliable, less sophisticated than hierarchical approach

CALLED BY:
----------
- app.py line 24: Main UI entry point for all queries

DEPENDENCIES:
-------------
- llama_index: Vector store abstractions
- pinecone: Vector database client
- groq: LLM client
- mixedbread: Neural reranking (optional)

DESIGN DECISIONS:
-----------------
1. WHY NOT RECURSIVE RETRIEVAL?:
   - Required complex local index persistence
   - Prone to storage corruption issues
   - Direct Pinecone query works reliably
   - Trade-off: Slightly less context-aware

2. WHY GLOBAL base_engine?:
   - Built once at module load (startup cost)
   - Reused across all queries (fast)
   - Maintains connection pools
   
3. WHY STREAMING DISABLED?:
   - Groq supports it, but adds complexity
   - Non-streaming gives full response at once
   - Easier to debug and test
   - app.py handles fake streaming for UX
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.nomic import NomicEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import settings
from utils import generate_hypothetical_answer, fetch_live_web_results

import os
os.environ["NOMIC_API_KEY"] = settings.nomic_api_key

Settings.embed_model = NomicEmbedding(
    model_name="nomic-embed-text-v1.5",
    api_key=settings.nomic_api_key
)

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
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)  # Immutable response container
class QueryResponseBundle:
    """
    Container for query results with metadata.
    
    DESIGN PATTERN: Value Object (immutable, pass-by-value semantics)
    
    PURPOSE:
    --------
    Bundle all query outputs into single object:
    - Main LLM response
    - Retrieved source documents
    - Confidence metrics
    - Fallback results (if triggered)
    
    WHY FROZEN?:
    ------------
    - Prevents accidental mutation: bundle.avg_score = 0.9 ❌ raises error
    - Thread-safe (multiple threads can read without locks)
    - Hashable (can use as dict key if needed)
    
    USAGE:
    ------
    bundle = query_engine("What is machine learning?")
    print(f"Confidence: {bundle.avg_score:.2f}")
    print(f"Answer: {bundle.response.response}")
    if bundle.low_confidence:
        print(f"Web fallback: {bundle.fallback_answer}")
    
    FIELDS:
    -------
    """
    
    response: Any  
    # TYPE: LlamaIndex Response object
    # CONTAINS:
    #   - .response: str (LLM-generated answer)
    #   - .source_nodes: List[NodeWithScore] (retrieved chunks)
    #   - .response_gen: Generator (for streaming, though we don't use it)
    # USAGE: response.response to get answer text
    # WHY Any?: LlamaIndex types are complex, Any simplifies typing
    
    avg_score: float
    # Confidence metric: average similarity score of top 6 retrieved chunks
    # RANGE: 0.0 (no relevant docs) to 1.0 (perfect match)
    # INTERPRETATION:
    #   - 0.9+: Excellent match, high confidence
    #   - 0.8-0.9: Good match, moderate confidence
    #   - 0.7-0.8: Weak match, low confidence
    #   - <0.7: Very weak or no match
    # CALCULATED: Line 87 in query_engine() function
    
    low_confidence: bool
    # Binary flag: avg_score < confidence_threshold (default 0.80)
    # TRUE triggers web fallback search
    # LOGIC: Line 88
    # USAGE: if bundle.low_confidence: show_warning()
    
    fallback_answer: Optional[str] = None
    # LLM-generated answer from web search results
    # POPULATED: Only if low_confidence=True AND Tavily API configured
    # FORMAT: Plain text string (200-500 words typically)
    # USAGE: Display below main answer with "Live Web Update" header
    # None if web fallback not triggered or API unavailable
    
    fallback_results: Optional[List[Dict[str, str]]] = None
    # Raw web search results from Tavily
    # STRUCTURE:
    #   [
    #       {"title": "...", "url": "...", "content": "..."},
    #       ...
    #   ]
    # USAGE: Display as source citations for fallback answer
    # None if no web search performed


# ============================================================================
# GLOBAL INITIALIZATION (Module Load Time)
# ============================================================================

# These objects are created ONCE when module is first imported
# WHY GLOBAL?:
# - Expensive to create (API connections, index loading)
# - Reused across all queries (connection pooling)
# - No state changes after initialization (thread-safe)

# ----------------------------------------------------------------------------
# 1. EMBEDDING MODEL
# ----------------------------------------------------------------------------

def _build_embed_model():
    if settings.embedding_model.startswith("nomic"):
        return NomicEmbedding(model_name=settings.embedding_model)
    return HuggingFaceEmbedding(model_name=settings.embedding_model)

Settings.embed_model = _build_embed_model()

# ----------------------------------------------------------------------------
# 2. LLM CLIENT
# ----------------------------------------------------------------------------

llm = Groq(
    model=settings.llm_model,
    api_key=settings.groq_api_key,
    temperature=0.1,
    max_retries=2,  # Retry failed requests (rate limits, network errors)
    timeout=30.0,   # 30 second timeout per request
)
# PURPOSE: Generate answers and hypothetical documents
# MODEL: "llama-3.3-70b-versatile" (70 billion parameters)
# TEMPERATURE: 0.1 = mostly deterministic, minimal creativity
#   - 0.0: Completely deterministic (same query → same answer)
#   - 0.1: Tiny variation (recommended for factual QA)
#   - 1.0: Creative (good for stories, bad for facts)
# MAX_RETRIES: 2 automatic retries on rate limits (default waits grow exponentially)
# TIMEOUT: 30s max per request (prevents infinite hangs)
# CONNECTION: Persistent HTTP/2 connection pool to api.groq.com
# USED BY:
#   - base_engine for answer synthesis
#   - generate_hypothetical_answer() in utils.py

# ----------------------------------------------------------------------------
# 3. PINECONE CONNECTION
# ----------------------------------------------------------------------------

pc = Pinecone(api_key=settings.pinecone_api_key)
# PURPOSE: Vector database client
# SINGLETON: Single client for all index operations
# THREAD-SAFE: Pinecone client handles concurrent queries
# COST: ~100ms to initialize gRPC connection

vector_store = PineconeVectorStore(pinecone_index=pc.Index(settings.index_name))
# PURPOSE: LlamaIndex wrapper around Pinecone index
# INDEX: Connects to specific index (e.g., "rag-domain-expert-2025")
# OPERATIONS:
#   - query(vector, top_k) → search for similar vectors
#   - upsert(vectors) → add/update vectors (used in ingest.py)
# CONTENTS: 537 vectors (from our ingestion) with metadata

# ----------------------------------------------------------------------------
# 4. QUERY ENGINE CONSTRUCTION
# ----------------------------------------------------------------------------

from llama_index.core import VectorStoreIndex, StorageContext

# Create storage context (connects vector store to LlamaIndex)
# NO LOCAL STORAGE: persist_dir not specified (ephemeral context)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
# CONTAINS:
#   - vector_store: Our Pinecone connection
#   - docstore: In-memory (empty, not used in simple mode)
#   - index_store: In-memory (empty, not used in simple mode)

# Build vector index from Pinecone
vector_index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)
# PURPOSE: LlamaIndex abstraction over raw Pinecone
# LOADS: Metadata about index, not the vectors themselves
# COST: ~50ms to fetch index stats

# ----------------------------------------------------------------------------
# 5. POSTPROCESSORS (Filtering & Reranking)
# ----------------------------------------------------------------------------

postprocessors = [SimilarityPostprocessor(similarity_cutoff=settings.similarity_cutoff)]
# STAGE 1: Similarity filtering
# PURPOSE: Remove low-relevance chunks before reranking
# CUTOFF: 0.77 (configurable via .env)
# LOGIC: if chunk.score < 0.77: discard
# WHY BEFORE RERANKING?: 
#   - Saves reranker API calls (only rerank relevant chunks)
#   - Reranker is slow, filter first

# Optional reranker (only if package installed)
try:
    from llama_index.postprocessor.mixedbread_rerank import MixedbreadAIRerank
    postprocessors.append(MixedbreadAIRerank(model=settings.reranker_model, top_n=6))
    # STAGE 2: Neural reranking
    # PURPOSE: Improve relevance scoring beyond cosine similarity
    # HOW?: Cross-encoder sees query + chunk together (vs separate embeddings)
    # MODEL: "mixedbread-ai/mxbai-rerank-base-v1"
    # TOP_N: Keep only top 6 after reranking
    # COST: ~100-200ms per query (6 chunks × ~20ms each)
    
    logger.info("Using Mixedbread AI reranker")
except ImportError:
    # Graceful degradation if package not installed
    # IMPACT: ~5-10% lower retrieval quality, but system still works
    # INSTALL: pip install llama-index-postprocessor-mixedbread-rerank
    logger.warning("Mixedbread AI reranker not available")

# ----------------------------------------------------------------------------
# 6. FINAL QUERY ENGINE
# ----------------------------------------------------------------------------

base_engine = vector_index.as_query_engine(
    similarity_top_k=20,  # Retrieve 20 chunks from Pinecone
    # WHY 20?: 
    #   - Cast wide net before filtering
    #   - After similarity_cutoff + reranking, ~6 remain
    #   - Too few (5): Might miss relevant docs
    #   - Too many (50): Slower, more noise
    
    node_postprocessors=postprocessors,  # Apply filtering + reranking
    # ORDER: [SimilarityPostprocessor, MixedbreadAIRerank]
    # PIPELINE: 20 chunks → 12 chunks (similarity) → 6 chunks (reranking)
    
    response_mode="compact",
    # STRATEGY: How to synthesize answer from multiple chunks
    # OPTIONS:
    #   - "refine": Iterative refinement (chunk by chunk) - SLOW, many LLM calls
    #   - "compact": Concatenate chunks, single LLM call - FAST, rate-limit friendly
    #   - "tree_summarize": Hierarchical summarization - BEST QUALITY but hits rate limits
    # COMPACT MODE:
    #   - Concatenates all chunks into one prompt
    #   - Single LLM call (avoids rate limits)
    #   - Good quality for 6 chunks (post-reranking)
    # TRADEOFF: Slightly lower quality than tree_summarize, but 3-5x fewer API calls

    streaming=False,  # Disable streaming for rate limit compliance
    # CRITICAL: streaming=True causes multiple API calls even in compact mode
    # - Each token chunk = separate API request
    # - Hits Groq rate limits (30 req/min)
    # - Causes 30-60 second delays per query
    # SOLUTION: streaming=False = truly single LLM call
    # NOTE: app.py fakes streaming in UI, so user experience unchanged
    
    llm=llm,  # Use our global Groq LLM instance
)

# RESULT: base_engine is ready to accept queries
# TYPE: RetrieverQueryEngine
# USAGE: base_engine.query("question") → Response object


# ============================================================================
# MAIN QUERY FUNCTION
# ============================================================================

def query_engine(question: str) -> QueryResponseBundle:
    """
    Execute full RAG pipeline with smart early-exit optimization.

    OPTIMIZATION STRATEGY:
    ----------------------
    1. Fast pre-check: Quick retrieval without HyDE to detect domain match
    2. If very low scores (<0.5) → Skip expensive steps, go straight to web
    3. If decent scores (≥0.5) → Run full pipeline with HyDE for best quality

    PERFORMANCE GAINS:
    ------------------
    - Out-of-domain queries: 2-3s (vs 5-7s before)
    - In-domain queries: 3-4s (unchanged, full quality)
    """

    logger.info("Incoming query: %s", question[:200])

    # ========================================================================
    # STAGE 1: FAST PRE-CHECK (Skip HyDE for initial probe)
    # ========================================================================
    # Do a quick retrieval with raw question to check domain relevance
    # COST: ~150ms (embed + Pinecone query, no LLM)

    retriever = vector_index.as_retriever(similarity_top_k=5)
    initial_nodes = retriever.retrieve(question)
    initial_scores = [node.score for node in initial_nodes if node.score is not None]
    max_initial_score = max(initial_scores) if initial_scores else 0.0

    logger.info("Pre-check max score: %.3f", max_initial_score)

    # ========================================================================
    # STAGE 2: EARLY EXIT FOR OUT-OF-DOMAIN QUERIES
    # ========================================================================
    # If best match is below similarity_cutoff, skip HyDE + synthesis, go straight to web
    # THRESHOLD: Use same cutoff as SimilarityPostprocessor (0.77 default)
    # RATIONALE: If pre-check can't even pass the filter, full pipeline won't help
    #   - Paris tourism in ArXiv: ~0.74 (fails cutoff)
    #   - Transformers in ArXiv: ~0.85+ (passes cutoff)
    #   - Saves: HyDE generation (1s) + synthesis (1-2s) = 2-3s total

    if max_initial_score < settings.similarity_cutoff:
        logger.warning("Very low pre-check score (%.3f), skipping HyDE and going straight to web", max_initial_score)

        # Check if Tavily API key configured
        if settings.tavily_api_key:
            # Fetch live web results
            fallback_results = fetch_live_web_results(question, settings.tavily_api_key)

            if fallback_results:
                # Format web results into prompt
                snippets = "\n\n".join(
                    f"Title: {item.get('title')}\nURL: {item.get('url')}\nSummary: {item.get('content')}"
                    for item in fallback_results
                )

                # Construct fallback prompt
                fallback_prompt = (
                    "The question appears to be outside the internal knowledge base. "
                    "Use the web search results below to answer factually with citations.\n\n"
                    f"{snippets}\n\nUser question: {question}\n\nAnswer:"
                )

                # Generate answer from web results
                fallback_answer = llm.complete(fallback_prompt).text.strip()
                logger.info("Produced web-only answer (early exit)")

                # Return web-only response (no internal RAG synthesis)
                dummy_response = Response(response=fallback_answer)

                return QueryResponseBundle(
                    response=dummy_response,
                    avg_score=max_initial_score,
                    low_confidence=True,
                    fallback_answer=fallback_answer,
                    fallback_results=fallback_results,
                )

        # No Tavily key or web search failed, return generic response
        logger.warning("No web fallback available for out-of-domain query")
        dummy_response = Response(
            response="I couldn't find relevant information in the knowledge base for this question. "
                    "This appears to be outside my domain expertise."
        )
        return QueryResponseBundle(
            response=dummy_response,
            avg_score=max_initial_score,
            low_confidence=True,
            fallback_answer=None,
            fallback_results=None,
        )

    # ========================================================================
    # STAGE 3: FULL PIPELINE FOR IN-DOMAIN QUERIES
    # ========================================================================
    # Score ≥ similarity_cutoff means we have relevant docs, use full HyDE pipeline

    logger.info("Pre-check passed (%.3f ≥ %.2f), running full HyDE pipeline", max_initial_score, settings.similarity_cutoff)

    # Generate hypothetical answer to improve retrieval
    hypo = generate_hypothetical_answer(llm, question)

    # Construct enhanced query by appending original question
    enhanced_query = f"{hypo}\n\nUsing the above as context, now answer precisely: {question}"

    # Query the engine with enhanced query
    response = base_engine.query(enhanced_query)

    # ========================================================================
    # STAGE 4: CONFIDENCE SCORING
    # ========================================================================

    scores = [node.score for node in response.source_nodes[:6] if node.score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    low_confidence = avg_score < settings.confidence_threshold

    logger.info("Query completed: avg_score=%.3f, source_nodes=%d", avg_score, len(response.source_nodes))

    # ========================================================================
    # STAGE 5: WEB FALLBACK (CONDITIONAL)
    # ========================================================================

    fallback_answer: Optional[str] = None
    fallback_results: Optional[List[Dict[str, str]]] = None

    if low_confidence:
        logger.warning("Low confidence detected, triggering web fallback")

        if settings.tavily_api_key:
            fallback_results = fetch_live_web_results(question, settings.tavily_api_key)

            if fallback_results:
                snippets = "\n\n".join(
                    f"Title: {item.get('title')}\nURL: {item.get('url')}\nSummary: {item.get('content')}"
                    for item in fallback_results
                )

                fallback_prompt = (
                    "Internal knowledge base confidence was low. "
                    "Use the external web snippets below to answer factually with citations when possible.\n\n"
                    f"{snippets}\n\nUser question: {question}\n\nAnswer:"
                )

                fallback_answer = llm.complete(fallback_prompt).text.strip()
                logger.info("Produced fallback answer using web snippets")

    return QueryResponseBundle(
        response=response,
        avg_score=avg_score,
        low_confidence=low_confidence,
        fallback_answer=fallback_answer,
        fallback_results=fallback_results,
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = ["query_engine", "QueryResponseBundle"]

"""
SYSTEM FLOW DIAGRAM:
====================

User Question
     ↓
query_engine(question)
     ↓
generate_hypothetical_answer() ←┐ utils.py
     ↓                           │
enhanced_query                   │
     ↓                           │
base_engine.query()              │
     ├→ Embed query              │
     ├→ Pinecone search (top 20) │
     ├→ Filter (similarity > 0.77)
     ├→ Rerank (top 6)           │
     ├→ LLM synthesize           │
     ↓                           │
Response + Scores                │
     ↓                           │
avg_score < 0.80? ──No──┐       │
     │                  │       │
     Yes                │       │
     ↓                  │       │
fetch_live_web_results()←┘       │ utils.py
     ↓                           │
Fallback Answer                  │
     ↓                           │
QueryResponseBundle              │
     ↓                           │
Display in UI                    │
"""
