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

llm = Groq(model=settings.llm_model, api_key=settings.groq_api_key, temperature=0.1)
# PURPOSE: Generate answers and hypothetical documents
# MODEL: "llama-3.3-70b-versatile" (70 billion parameters)
# TEMPERATURE: 0.1 = mostly deterministic, minimal creativity
#   - 0.0: Completely deterministic (same query → same answer)
#   - 0.1: Tiny variation (recommended for factual QA)
#   - 1.0: Creative (good for stories, bad for facts)
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
    
    response_mode="tree_summarize",
    # STRATEGY: How to synthesize answer from multiple chunks
    # OPTIONS:
    #   - "refine": Iterative refinement (chunk by chunk)
    #   - "compact": Concatenate chunks, single LLM call
    #   - "tree_summarize": Hierarchical summarization (best for >3 chunks)
    # TREE_SUMMARIZE:
    #   1. Summarize pairs of chunks
    #   2. Summarize summaries
    #   3. Final synthesis
    # TRADEOFF: More LLM calls, but better quality for many chunks
    
    streaming=True,  # Enable token-by-token generation
    # NOTE: We don't actually use streaming (app.py fakes it)
    # KEPT: For future enhancement or debugging
    # ALTERNATIVE: streaming=False for simpler responses
    
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
    Execute full RAG pipeline for a user question.
    
    PIPELINE STAGES:
    ----------------
    1. Log incoming query
    2. Generate HyDE hypothesis
    3. Query vector store (retrieve relevant chunks)
    4. Calculate confidence scores
    5. Generate answer from retrieved chunks
    6. [Optional] Web fallback if low confidence
    7. Return bundled results
    
    PARAMETERS:
    -----------
    question : str
        User's natural language question
        EXAMPLES:
            - "What is machine learning?"
            - "How does Pinecone work?"
            - "Recent papers on transformers?"
        LENGTH: Usually 5-50 words
        
    RETURNS:
    --------
    QueryResponseBundle : Complete query results
        See QueryResponseBundle docstring for field details
        
    PERFORMANCE:
    ------------
    Typical latency breakdown (milliseconds):
        HyDE generation: 500-1000ms
        Pinecone retrieval: 50-100ms
        Reranking: 100-200ms
        Answer synthesis: 1000-2000ms
        [Web fallback: +2000-3000ms if triggered]
        ---
        Total: 1.5-3.5 seconds (without fallback)
        
    CALLED BY:
    ----------
    - app.py line 24: Every user query from UI
    
    ERROR HANDLING:
    ---------------
    - Network failures: LlamaIndex retries automatically
    - Rate limits: Groq client handles backoff
    - Empty results: Returns low confidence + web fallback
    - LLM errors: Propagates exception (no graceful degradation)
    
    THREAD SAFETY:
    --------------
    - SAFE: Multiple concurrent calls OK
    - base_engine handles internal locking
    - Pinecone client is thread-safe
    - Groq client uses connection pooling
    """
    
    # ========================================================================
    # STAGE 1: LOGGING & SETUP
    # ========================================================================
    
    logger.info("Incoming query: %s", question[:200])
    # WHY [:200]?: Truncate long questions to keep logs readable
    # LEVEL: INFO (always logged, even in production)
    # OUTPUT: "2025-11-25 12:00:00 INFO [query_engine] Incoming query: What is..."
    
    # ========================================================================
    # STAGE 2: HYDE QUERY ENHANCEMENT
    # ========================================================================
    
    # Generate hypothetical answer to improve retrieval
    hypo = generate_hypothetical_answer(llm, question)
    # COST: 1 LLM call (~500-1000ms)
    # OUTPUT: "Machine learning is a subset of AI that enables systems..."
    # LENGTH: Typically 100-500 words
    
    # Construct enhanced query by appending original question
    enhanced_query = f"{hypo}\n\nUsing the above as context, now answer precisely: {question}"
    # STRUCTURE:
    #   [Hypothetical answer]
    #   
    #   Using the above as context, now answer precisely: [Original question]
    #
    # WHY THIS FORMAT?:
    #   - Hypo provides vocabulary for retrieval
    #   - Original question guides answer generation
    #   - "Using above as context" tells LLM to reference hypothesis
    
    # ========================================================================
    # STAGE 3: RETRIEVAL & ANSWER GENERATION
    # ========================================================================
    
    # Query the engine with enhanced query
    response = base_engine.query(enhanced_query)
    # WHAT HAPPENS INTERNALLY:
    #   1. Embed enhanced_query → 768-dim vector
    #   2. Query Pinecone for top_k=20 similar chunks
    #   3. Apply SimilarityPostprocessor (filter < 0.77)
    #   4. Apply MixedbreadAIRerank (keep top 6)
    #   5. Pass chunks + question to LLM
    #   6. LLM synthesizes answer using tree_summarize
    #   7. Return Response object
    #
    # RESPONSE STRUCTURE:
    #   - .response: str (final answer)
    #   - .source_nodes: List[NodeWithScore] (chunks used)
    #   - .response_gen: Generator (for streaming)
    
    # ========================================================================
    # STAGE 4: CONFIDENCE SCORING
    # ========================================================================
    
    # Extract similarity scores from top 6 retrieved chunks
    scores = [node.score for node in response.source_nodes[:6] if node.score is not None]
    # WHY [:6]?: Reranker returned top 6, focus on those
    # WHY if node.score is not None?: Defensive programming (scores should always exist)
    # OUTPUT: [0.85, 0.83, 0.81, 0.79, 0.77, 0.75] (example)
    
    # Calculate average confidence score
    avg_score = sum(scores) / len(scores) if scores else 0.0
    # LOGIC: Mean of top 6 scores
    # EDGE CASE: if scores empty (no results) → 0.0
    # OUTPUT: 0.800 (example)
    
    # Determine if confidence is below threshold
    low_confidence = avg_score < settings.confidence_threshold  # 0.80 default
    # BOOLEAN: True if avg_score < 0.80
    # TRIGGERS: Web fallback if True
    
    logger.info("Query completed: avg_score=%.3f, source_nodes=%d", avg_score, len(response.source_nodes))
    # LOG OUTPUT: "Query completed: avg_score=0.800, source_nodes=6"
    # PURPOSE: Track retrieval quality in logs
    
    # ========================================================================
    # STAGE 5: WEB FALLBACK (CONDITIONAL)
    # ========================================================================
    
    fallback_answer: Optional[str] = None
    fallback_results: Optional[List[Dict[str, str]]] = None
    # INITIALIZE: Both None (no fallback by default)
    
    if low_confidence:
        # Confidence below threshold → internal knowledge insufficient
        logger.warning("Low confidence detected, triggering web fallback")
        
        # Check if Tavily API key configured
        if settings.tavily_api_key:
            # Fetch live web results
            fallback_results = fetch_live_web_results(question, settings.tavily_api_key)
            # COST: 1 API call (~1-2 seconds)
            # OUTPUT: [{"title": "...", "url": "...", "content": "..."}, ...]
            
            if fallback_results:
                # Format web results into prompt
                snippets = "\n\n".join(
                    f"Title: {item.get('title')}\nURL: {item.get('url')}\nSummary: {item.get('content')}"
                    for item in fallback_results
                )
                # EXAMPLE:
                #   Title: Quantum Computing Explained
                #   URL: https://ibm.com/quantum
                #   Summary: Quantum computers use qubits...
                #
                #   Title: Introduction to Qubits
                #   URL: https://nature.com/qubits
                #   Summary: Qubits can exist in superposition...
                
                # Construct fallback prompt
                fallback_prompt = (
                    "Internal knowledge base confidence was low. "
                    "Use the external web snippets below to answer factually with citations when possible.\n\n"
                    f"{snippets}\n\nUser question: {question}\n\nAnswer:"
                )
                # INSTRUCTION: Use web results to supplement internal knowledge
                # REQUIREMENT: "with citations" encourages mentioning sources
                
                # Generate answer from web results
                fallback_answer = llm.complete(fallback_prompt).text.strip()
                # COST: 1 additional LLM call (~1-2 seconds)
                # OUTPUT: "According to IBM's quantum computing guide..."
                
                logger.info("Produced fallback answer using web snippets")
    
    # ========================================================================
    # STAGE 6: RETURN RESULTS
    # ========================================================================
    
    return QueryResponseBundle(
        response=response,  # Main answer + source nodes
        avg_score=avg_score,  # Confidence metric
        low_confidence=low_confidence,  # Boolean flag
        fallback_answer=fallback_answer,  # Web answer (or None)
        fallback_results=fallback_results,  # Web sources (or None)
    )
    # USAGE IN app.py:
    #   bundle = query_engine("What is X?")
    #   st.write(bundle.response.response)  # Main answer
    #   if bundle.low_confidence:
    #       st.warning(f"Low confidence: {bundle.avg_score}")
    #       if bundle.fallback_answer:
    #           st.write(bundle.fallback_answer)  # Web answer


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
