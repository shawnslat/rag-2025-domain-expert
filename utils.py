"""
Utility Functions Module
=========================
RAG 2025 Domain Expert - Advanced Retrieval Utilities

AUTHOR: Shawn Slattery
GitHub: https://github.com/shawnslat
LinkedIn: https://www.linkedin.com/in/shawn-slattery-843654201/

Built: November 2025
Features: HyDE (Hypothetical Document Embeddings), Web Fallback, Golden Set Evaluation

PURPOSE:
--------
Reusable helper functions for:
1. HyDE (Hypothetical Document Embeddings) - Query enhancement
2. Web fallback - Live search when internal knowledge insufficient
3. Evaluation - Testing system accuracy with golden sets

ARCHITECTURE:
-------------
Pure functions with no global state (except logger)
- Easy to test in isolation
- Can be imported and reused across modules
- Each function has a single, clear responsibility

CALLED BY:
----------
- query_engine.py: Uses all three functions for retrieval pipeline
- ingest.py (potential): Could use evaluate_with_golden_set for CI/CD

DEPENDENCIES:
-------------
- tenacity: Retry logic with exponential backoff
- requests: HTTP client for Tavily API
- llama_index: LLM interface for HyDE generation
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import logging
import requests
from llama_index.core.llms import LLM
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Check if logging is already configured (prevents duplicate handlers)
# WHY?: If multiple modules import utils, we don't want multiple log outputs
# PATTERN: Defensive check before basicConfig
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        # FORMAT: "2025-11-25 12:00:00 INFO [utils] Message here"
        #         timestamp        level  module   content
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

# Module-specific logger (shows up as [utils] in logs)
# USAGE: logger.info("message") vs print("message")
# WHY LOGGER?: 
#   - Configurable verbosity (DEBUG, INFO, WARNING)
#   - Timestamps and module names automatically added
#   - Can redirect to files, not just stdout
logger = logging.getLogger(__name__)  # __name__ = "utils" when imported


# ============================================================================
# HYDE (HYPOTHETICAL DOCUMENT EMBEDDINGS)
# ============================================================================

def generate_hypothetical_answer(llm: LLM, question: str) -> str:
    """
    Generate a hypothetical answer to improve retrieval accuracy.
    
    PROBLEM SOLVED:
    ---------------
    User questions and document text use different vocabulary:
    
    Query: "How do I fix memory leaks?"
    ↓ (direct embedding)
    Embedding: [0.1, -0.3, 0.5, ...]
    
    Document: "Memory leaks occur when allocated memory is not freed. 
               Use tools like Valgrind to detect leaks..."
    ↓ (document embedding)  
    Embedding: [0.4, 0.2, -0.1, ...]
    
    LOW SIMILARITY despite being highly relevant!
    
    HYDE SOLUTION:
    --------------
    1. LLM generates hypothetical answer (even if wrong)
    2. Hypothetical answer uses vocabulary similar to real docs
    3. Embed hypothetical answer instead of query
    4. Better similarity match with real documents
    
    EXAMPLE:
    --------
    Query: "How do I fix memory leaks?"
    ↓ (HyDE generates)
    Hypothesis: "Memory leaks can be fixed by ensuring all allocated 
                 memory is properly freed using free() or delete. Tools 
                 like Valgrind help detect leaks by tracking allocations..."
    ↓ (embed hypothesis)
    Embedding: [0.39, 0.18, -0.09, ...]  ← Much closer to document!
    
    PARAMETERS:
    -----------
    llm : LLM
        Language model interface (typically Groq Llama)
        SOURCE: query_engine.py line 37
    question : str
        User's original question
        
    RETURNS:
    --------
    str : Hypothetical answer text (100-500 words typically)
    
    CALLED BY:
    ----------
    - query_engine.py line 74: Before embedding query
    
    RESEARCH:
    ---------
    Gao et al. (2022): "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496
    
    COST:
    -----
    - 1 LLM call per query (~100-200 tokens generated)
    - Adds ~500ms-1s latency (Groq is fast)
    - Worth it: Typical improvement of 10-20% in retrieval accuracy
    
    PROMPT ENGINEERING:
    -------------------
    - "world-class researcher" → Encourages technical, detailed language
    - "detailed, factual" → Prevents vague or creative responses
    - "as if you had perfect knowledge" → Removes hedging language
    """
    # Carefully crafted prompt to elicit document-like language
    # Each phrase has a purpose (not arbitrary)
    prompt = (
        "You are a world-class researcher in this domain. "  # Authority/expertise signal
        "Write a detailed, factual hypothetical answer to the question below "  # Style instruction
        "as if you had perfect knowledge.\n\n"  # Remove uncertainty hedging
        f"Question: {question}\n\nAnswer:"  # Clear structure
    )
    
    # LLM.complete() is synchronous (blocks until response)
    # ALTERNATIVE: Could use acomplete() for async if needed
    # .text extracts string from CompletionResponse object
    return llm.complete(prompt).text
    # RETURNS: "Memory leaks occur when..."
    # LENGTH: Typically 100-500 words (controlled by prompt implicitly)


# ============================================================================
# EVALUATION HARNESS
# ============================================================================

def evaluate_with_golden_set(
    golden_pairs: Sequence[tuple[str, str]],
    ask_fn,
) -> list[dict]:
    """
    Automated testing framework for RAG system accuracy.
    
    PURPOSE:
    --------
    Prevent silent degradation of RAG system quality:
    - Index corruption → Returns wrong docs
    - API changes → Parsing breaks
    - New data → Dilutes relevance
    - Model updates → Different behavior
    
    GOLDEN SET CONCEPT:
    -------------------
    Curated set of (question, expected_answer) pairs:
    
    [
        ("What is the capital of France?", "Paris"),
        ("When was Python created?", "1991"),
        ("Who invented transformers?", "Vaswani"),
    ]
    
    RUN PERIODICALLY:
    -----------------
    - Before deployments (CI/CD pipeline)
    - Nightly cron jobs
    - After index updates
    - After model changes
    
    ALERTING:
    ---------
    if accuracy < 0.80:  # Threshold configurable
        send_alert("RAG system degraded!")
        block_deployment()
    
    PARAMETERS:
    -----------
    golden_pairs : Sequence[tuple[str, str]]
        List of (question, reference_answer) tuples
        EXAMPLE:
            [
                ("What is HNSW?", "Hierarchical Navigable Small World"),
                ("Pinecone uses what algorithm?", "approximate nearest neighbor"),
            ]
        
    ask_fn : Callable[[str], Any]
        Function that takes a question and returns an answer
        SIGNATURE: (question: str) -> str | QueryResponseBundle | Response
        EXAMPLE: lambda q: query_engine(q)
        
    RETURNS:
    --------
    list[dict] : Evaluation results
        STRUCTURE:
            [
                {
                    "question": "What is HNSW?",
                    "answer": "HNSW stands for Hierarchical...",
                    "reference": "Hierarchical Navigable Small World",
                    "match": True  # reference substring found in answer
                },
                ...
            ]
    
    CALLED BY:
    ----------
    - Could be used in: tests/test_rag_accuracy.py
    - Could be used in: scripts/evaluate_before_deploy.sh
    - Currently: Manual testing only (no CI/CD integration yet)
    
    MATCHING LOGIC:
    ---------------
    Simple substring matching (case-insensitive)
    LIMITATION: 
        - "Paris" in "The capital is Paris, France" → ✅ Match
        - "1991" in "Python 2 was released in 2000" → ❌ False positive
    
    FUTURE IMPROVEMENTS:
    --------------------
    - Use LLM-as-judge: "Does this answer correctly answer the question?"
    - Use semantic similarity: cosine(embed(answer), embed(reference))
    - Use exact match with normalization: strip punctuation, lowercase
    """
    
    # Inner helper function to extract text from various response types
    # WHY INNER FUNCTION?: 
    #   - Only used here, no need to pollute module namespace
    #   - Closures have access to parent scope if needed
    def _extract_answer_text(raw: Any) -> str:
        """
        Normalize different response types to plain text.
        
        HANDLES:
        --------
        1. Plain string: "The answer is Paris"
        2. QueryResponseBundle: bundle.response.response
        3. LlamaIndex Response: response.response
        4. CompletionResponse: response.text
        5. None: ""
        
        WHY COMPLEX?:
        -------------
        Different modules return different types:
        - Direct LLM: CompletionResponse
        - Query engine: QueryResponseBundle
        - Simple wrapper: str
        
        DEFENSIVE PROGRAMMING:
        ----------------------
        Try multiple extraction paths, fallback to str(raw) if all fail
        NEVER raises exception (returns "" or str(raw))
        """
        # Type guards in order of likelihood (most common first)
        
        # Case 1: None or empty
        if raw is None:
            return ""
        
        # Case 2: Already a string (simplest case)
        if isinstance(raw, str):
            return raw
        
        # Case 3: QueryResponseBundle with fallback answer
        # PRIORITY: Fallback is more recent/relevant than base response
        if hasattr(raw, "fallback_answer") and getattr(raw, "fallback_answer"):
            return raw.fallback_answer  # type: ignore[attr-defined]
        
        # Case 4: Try to get .response attribute
        # Works for QueryResponseBundle, Response, etc.
        candidate = getattr(raw, "response", raw)
        if isinstance(candidate, str):
            return candidate
        
        # Case 5: Nested .response.response (some LlamaIndex types)
        if hasattr(candidate, "response") and isinstance(candidate.response, str):  # type: ignore[attr-defined]
            return candidate.response  # type: ignore[attr-defined]
        
        # Case 6: .text attribute (CompletionResponse)
        if hasattr(candidate, "text"):
            return candidate.text  # type: ignore[attr-defined]
        
        # Case 7: Give up, convert to string
        # LAST RESORT: Better than crashing
        return str(raw)

    # Main evaluation loop
    evaluations: list[dict] = []  # Type hint for return value
    
    for question, reference in golden_pairs:
        # Execute query (could take 1-5 seconds depending on system)
        # ERROR HANDLING: Could wrap in try/except to catch failures
        answer_raw = ask_fn(question)
        
        # Normalize to text for comparison
        answer_text = _extract_answer_text(answer_raw)
        
        # Case-insensitive substring matching
        # LOGIC: "paris" in "The capital is Paris, France".lower()
        # LIMITATION: Doesn't handle synonyms ("US" vs "United States")
        match = reference.lower() in answer_text.lower()
        
        # Store structured results
        evaluations.append(
            {
                "question": question,
                "answer": answer_text,
                "reference": reference,
                "match": match,  # Boolean: Did we find the reference?
            }
        )
    
    return evaluations
    # EXAMPLE USAGE:
    #   results = evaluate_with_golden_set(golden, query_engine)
    #   accuracy = sum(r["match"] for r in results) / len(results)
    #   print(f"Accuracy: {accuracy:.1%}")  # "Accuracy: 85.0%"


# ============================================================================
# WEB FALLBACK - LIVE SEARCH
# ============================================================================

def fetch_live_web_results(
    query: str,
    api_key: str,
    max_results: int = 4,
) -> List[Dict[str, str]]:
    """
    Query Tavily API for live web search when internal knowledge insufficient.
    
    FALLBACK STRATEGY:
    ------------------
    When RAG system confidence < threshold:
    1. Internal docs don't have answer
    2. Query Tavily for fresh web content
    3. Synthesize answer from web + internal knowledge
    4. Show user both sources
    
    WHY TAVILY?:
    ------------
    Optimized for LLM use cases vs Google/Bing:
    - Returns clean text snippets (not HTML)
    - Structured JSON (easy to parse)
    - "search_depth" modes for quick vs comprehensive
    - Reasonable pricing ($1 per 1000 searches)
    
    ALTERNATIVES:
    -------------
    - Exa.ai: Semantic search, better for research
    - Brave Search API: Privacy-focused, cheaper
    - Google Custom Search: More results, harder to parse
    - Bing Search API: Microsoft ecosystem integration
    
    PARAMETERS:
    -----------
    query : str
        User's question (same as RAG query)
        EXAMPLE: "What is quantum computing?"
        
    api_key : str
        Tavily API key from config.settings.tavily_api_key
        FORMAT: "tvly-..."
        GET IT: https://tavily.com/api-keys
        
    max_results : int
        Number of web results to return (default: 4)
        TRADEOFF: 
            - More results → Better coverage, slower, more LLM tokens
            - Fewer results → Faster, cheaper, may miss relevant info
        RECOMMENDATION: 3-5 for most use cases
    
    RETURNS:
    --------
    List[Dict[str, str]] : Cleaned search results
        STRUCTURE:
            [
                {
                    "title": "Quantum Computing Explained",
                    "url": "https://example.com/quantum",
                    "content": "Quantum computing uses qubits to..."
                },
                ...
            ]
        
        ON ERROR: Returns [] (empty list)
            - API key missing
            - Rate limit hit
            - Network error
            - Invalid response
    
    CALLED BY:
    ----------
    - query_engine.py line 93-105: Low confidence fallback
    
    ERROR HANDLING:
    ---------------
    Graceful degradation philosophy:
    - Network failure? Return []
    - Rate limit? Return [] after retries
    - Invalid JSON? Return []
    - User sees: "Low confidence, no web results" (not a crash)
    
    RETRY LOGIC:
    ------------
    Uses tenacity for exponential backoff:
    - Attempt 1: Immediate
    - Attempt 2: Wait 2 seconds
    - Attempt 3: Wait 4 seconds
    - Give up: Return []
    
    Total max wait: ~6 seconds
    """
    
    # Early exit if no API key configured
    # PHILOSOPHY: Fail fast with clear error vs cryptic HTTP 401
    if not api_key:
        return []  # Empty results, not an exception
    
    # Construct Tavily API request payload
    # DOCS: https://docs.tavily.com/docs/python-sdk/tavily-search/api-reference
    payload: Dict[str, Any] = {
        "api_key": api_key,  # Authentication
        "query": query,  # Search terms
        
        # "advanced" mode:
        # - Follows links to get more context
        # - ~2-3x slower than "basic"
        # - Better for complex queries
        # ALTERNATIVE: "basic" for speed (1-2 results, surface-level)
        "search_depth": "advanced",
        
        "max_results": max_results,  # Cap number of results
    }
    
    # ========================================================================
    # RETRY DECORATOR - Exponential Backoff
    # ========================================================================
    
    @retry(
        # Stop after 3 attempts (1 initial + 2 retries)
        stop=stop_after_attempt(3),
        
        # Exponential backoff: 2s, 4s, 8s (multiplier=1 means powers of 2)
        # min=2: First retry waits at least 2 seconds
        # max=10: Never wait more than 10 seconds
        # FORMULA: min(max, multiplier * 2^(attempt_number - 1))
        wait=wait_exponential(multiplier=1, min=2, max=10),
        
        # Only retry on network errors, not 404s or bad requests
        # RequestException covers: Timeout, ConnectionError, HTTPError
        retry=retry_if_exception_type(requests.RequestException),
        
        # Re-raise exception after all retries exhausted
        # ALTERNATIVE: reraise=False would return None
        reraise=True,
        
        # Log before sleeping (helps debugging)
        # EXAMPLE LOG: "Retrying in 2 seconds: Connection timeout"
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_tavily() -> requests.Response:
        """
        Inner function that actually makes HTTP request.
        
        WHY INNER FUNCTION?:
        - Retry decorator needs a function to wrap
        - Keeps retry logic separate from parsing logic
        - Makes testing easier (can mock this function)
        
        RETURNS:
        --------
        requests.Response: HTTP response object
            - .json(): Parse JSON body
            - .status_code: HTTP status (200, 429, 500, etc.)
            - .raise_for_status(): Raise exception if 4xx or 5xx
        
        RAISES:
        -------
        requests.Timeout: Request took too long (15s timeout)
        requests.ConnectionError: Network unreachable
        requests.HTTPError: 4xx or 5xx status code
        """
        return requests.post(
            "https://api.tavily.com/search",  # Tavily API endpoint
            json=payload,  # Auto-converts dict to JSON + sets Content-Type header
            timeout=15,  # 15 second timeout (prevents infinite hang)
        )
    
    # Execute request with retry logic
    try:
        response = _call_tavily()  # May retry up to 3 times
        response.raise_for_status()  # Raise exception if HTTP error
        
    except requests.RequestException as exc:
        # Catch ALL network-related errors
        # - Timeout: Took too long
        # - ConnectionError: Network down
        # - HTTPError: 429 (rate limit), 500 (server error), etc.
        
        # Log the failure (helps debugging)
        logger.warning("Tavily request failed after retries: %s", exc)
        
        # Return empty results (graceful degradation)
        return []
    
    # ========================================================================
    # PARSE AND CLEAN RESULTS
    # ========================================================================
    
    # Parse JSON response
    # STRUCTURE: {"results": [{...}, {...}], "query": "...", ...}
    data = response.json()
    results = data.get("results") or []  # Use [] if "results" key missing
    
    # Clean and normalize results
    cleaned: List[Dict[str, str]] = []
    
    for item in results[:max_results]:  # Enforce max_results cap
        # Defensive field extraction with fallbacks
        # WHY?: Tavily schema could change, or fields could be null
        cleaned.append(
            {
                # Title: Try multiple sources, fallback to URL or "Result"
                "title": item.get("title") or item.get("url") or "Result",
                
                # URL: Always should exist, but be defensive
                "url": item.get("url") or "",
                
                # Content: Main text snippet
                # "snippet" is shorter version, "content" is full
                # PREFERENCE: content (more context) > snippet
                "content": item.get("content") or item.get("snippet") or "",
            }
        )
    
    return cleaned
    # EXAMPLE OUTPUT:
    # [
    #     {
    #         "title": "What is Quantum Computing?",
    #         "url": "https://ibm.com/quantum",
    #         "content": "Quantum computing harnesses quantum mechanics..."
    #     },
    #     ...
    # ]


# ============================================================================
# MODULE EXPORTS
# ============================================================================

# Explicitly declare public API (optional but good practice)
# USAGE: from utils import *  (imports only these functions)
# NOTE: Better to use explicit imports: from utils import generate_hypothetical_answer
__all__ = [
    "generate_hypothetical_answer",
    "evaluate_with_golden_set", 
    "fetch_live_web_results"
]

# TESTING NOTES:
# --------------
# 1. generate_hypothetical_answer:
#    - Mock llm.complete() to return fixed string
#    - Test prompt construction
#    
# 2. evaluate_with_golden_set:
#    - Test with various response types (str, None, objects)
#    - Test matching logic with edge cases
#    
# 3. fetch_live_web_results:
#    - Mock requests.post() to avoid real API calls
#    - Test retry logic by raising RequestException
#    - Test empty/malformed responses