"""
Streamlit Web Application Module
=================================
RAG 2025 Domain Expert - Interactive Chat Interface

AUTHOR: Shawn Slattery
GitHub: https://github.com/shawnslat
LinkedIn: https://www.linkedin.com/in/shawn-slattery-843654201/

Built: November 2025
Stack: Streamlit + LlamaIndex + Pinecone + Groq

PURPOSE:
--------
Production-ready chat interface for RAG system with:
- Streaming responses for real-time feedback
- Source citation with clickable links
- Confidence scoring and web fallback visualization
- Rate limiting to prevent abuse
- Session state management
- Expandable source inspection

FEATURES:
---------
1. Real-time streaming: Words appear as generated
2. Source attribution: Every answer cites documents
3. Web fallback: Auto-search when low confidence
4. Rate limiting: 5-second cooldown between queries
5. Storage monitoring: Shows last index update time
6. Analytics opt-in: Optional usage tracking

USAGE:
------
# Local development
streamlit run app.py

# Production deployment (Streamlit Cloud)
1. Push to GitHub
2. Connect at share.streamlit.io
3. Add secrets in dashboard
4. Deploy!

URL: http://localhost:8501

DEPENDENCIES:
-------------
- streamlit: Web framework
- query_engine: Core RAG pipeline
- config: Environment variables

PERFORMANCE:
------------
- Initial load: ~2-3s (model initialization)
- Per query: 1.5-3.5s (depends on complexity)
- Streaming: ~50-100 tokens/sec
- Memory: ~500MB (models loaded in memory)

DEPLOYMENT:
-----------
Streamlit Cloud (Free):
- Auto-deploys from GitHub
- Secrets management built-in
- SSL certificate included
- Custom domain support

CALLED BY:
----------
- User browser → http://localhost:8501
- Streamlit Cloud → Production URL
"""

import os
import time
from datetime import datetime

import streamlit as st

from config import settings
from query_engine import QueryResponseBundle, query_engine

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG 2025 Expert | Built by Shawn Slattery",  # Browser tab title
    # VISIBLE: In browser tab, bookmarks, search results
    
    layout="centered",  # Layout mode
    # OPTIONS:
    # - "centered": 730px max width (better for reading)
    # - "wide": Full browser width (better for dashboards)
    # CHOSEN: Centered for chat-style interface
    
    page_icon="🔥",  # Emoji or image URL
    # VISIBLE: Browser tab, bookmarks
    # BRANDING: Fire emoji = "hot" content, cutting edge
)

# ============================================================================
# SOCIAL MEDIA PREVIEW (OPEN GRAPH)
# ============================================================================

# OpenGraph tags for social media sharing
# SHOWN: When URL shared on LinkedIn, Twitter, Slack, etc.
st.markdown(
    """
    <meta property="og:title" content="RAG 2025 Domain Expert | Built by Shawn Slattery" />
    <meta property="og:description" content="Production-grade RAG with HyDE + Neural Reranking + Web Fallback" />
    <meta property="og:url" content="https://github.com/shawnslat/rag-2025-domain-expert" />
    <meta property="og:type" content="website" />
    """,
    unsafe_allow_html=True,
)
# PREVIEW: Shows card with title + description when shared
# BRANDING: Your name in the title for visibility


# ============================================================================
# HEADER AND BRANDING
# ============================================================================

st.title("🔥 Domain Expert RAG – Nov 2025 Edition")
# MAIN HEADING: First thing users see
# BRANDING: Date shows it's current, not a stale demo

st.caption("Llama-3.3-70B @ Groq • Nomic • Pinecone Serverless • HyDE + Mixedbread rerank")
# TECH STACK: Shows sophistication + what's under the hood
# APPEALS TO: Technical recruiters, ML engineers

st.caption("Built by [Shawn Slattery](https://github.com/shawnslat) | [LinkedIn](https://www.linkedin.com/in/shawn-slattery-843654201/)")
# ATTRIBUTION: Clickable links to your profiles
# VISIBILITY: Every user sees your name + links

st.markdown(
    "Precision answers with citations. Ask about the knowledge base and watch the system retrieve, rerank, and synthesize in real-time."
)
# VALUE PROPOSITION: What users can expect
# KEYWORDS: "Precision", "citations", "real-time" (professional language)


# ============================================================================
# API KEY VALIDATION
# ============================================================================

# Check for required API keys before doing anything
# PHILOSOPHY: Fail fast with clear error message
missing_keys = [
    name for name, val in {
        "GROQ_API_KEY": settings.groq_api_key,
        "PINECONE_API_KEY": settings.pinecone_api_key,
        "FIRECRAWL_API_KEY": settings.firecrawl_api_key,
        "NOMIC_API_KEY": os.getenv("NOMIC_API_KEY", "").strip(),
    }.items() if not val
]
# CHECKS: Each API key is non-empty
# RESULT: List of missing key names

if missing_keys:
    # CRITICAL ERROR: Can't run without these
    st.error(
        f"❌ Missing API keys: {', '.join(missing_keys)}. "
        "Fill them in `.env` and restart the app."
    )
    # USER GUIDANCE: Exactly what to do
    # FORMAT: Clear, actionable message
    
    st.stop()
    # HALT: Don't continue execution
    # PREVENTS: Cryptic errors later in the code


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Streamlit reruns entire script on every interaction
# Session state persists data across reruns
# PATTERN: Initialize once, then reuse

if "messages" not in st.session_state:
    st.session_state.messages = []
    # STORES: List of {role: "user"/"assistant", content: str, sources: list}
    # PURPOSE: Chat history for display
    # CLEARED: On page refresh or session timeout

if "last_query_ts" not in st.session_state:
    st.session_state.last_query_ts = 0.0
    # STORES: Unix timestamp of last query
    # PURPOSE: Rate limiting (prevent spam)
    # RESET: On page refresh

if "analytics_opt_in" not in st.session_state:
    st.session_state.analytics_opt_in = False
    # STORES: User consent for analytics
    # PURPOSE: Privacy compliance (GDPR-friendly)
    # DEFAULT: False (opt-in, not opt-out)


# ============================================================================
# CHAT HISTORY DISPLAY
# ============================================================================

# Display all previous messages
# RUNS: On every rerun, rebuilds UI from session state
for msg in st.session_state.messages:
    # STRUCTURE: {role: str, content: str, sources: list}
    
    with st.chat_message(msg["role"]):
        # CREATES: Chat bubble (user or assistant style)
        # STYLING: Different colors/icons per role
        
        st.markdown(msg["content"])
        # RENDERS: Message text with markdown formatting
        # SUPPORTS: Bold, italic, links, code blocks
        
        if msg.get("sources"):
            # OPTIONAL: Sources list if present
            st.caption("Sources: " + " • ".join(msg["sources"]))
            # FORMAT: Compact citation list
            # SEPARATOR: Bullet point (•) for readability


# ============================================================================
# RATE LIMITING
# ============================================================================

# Chat input widget (returns None until user submits)
if prompt := st.chat_input("Ask about the knowledge base…"):
    # WALRUS OPERATOR: Assigns and checks in one line
    # prompt = st.chat_input(...)
    # if prompt: ...
    
    # Check rate limit (5-second cooldown)
    now = time.time()  # Current Unix timestamp
    if now - st.session_state.last_query_ts < 5:
        # TOO FAST: Less than 5 seconds since last query
        st.warning("Taking a breather—please wait a few seconds between questions.")
        # USER FEEDBACK: Friendly message, not harsh error
        
        st.stop()
        # HALT: Don't process query
        # PREVENTS: API rate limits, abuse, high costs
    
    # Update timestamp (rate limit passed)
    st.session_state.last_query_ts = now
    
    
    # ========================================================================
    # USER MESSAGE DISPLAY
    # ========================================================================
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # STORES: For display on next rerun
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    # FEEDBACK: User sees their message right away
    
    
    # ========================================================================
    # ASSISTANT RESPONSE GENERATION
    # ========================================================================
    
    with st.chat_message("assistant"):
        # CREATES: Assistant chat bubble
        
        # Execute RAG pipeline
        bundle: QueryResponseBundle = query_engine(prompt)
        # DURATION: 1.5-3.5 seconds
        # RETURNS: Complete response bundle with metadata
        # COMPONENTS:
        # - response: Response object with streaming
        # - avg_score: Confidence (0.0-1.0)
        # - low_confidence: Boolean flag
        # - fallback_answer: Optional web search result
        # - fallback_results: Optional web search metadata
        
        response = bundle.response
        # EXTRACTS: Response object from bundle
        
        
        # ====================================================================
        # STREAMING DISPLAY
        # ====================================================================
        
        placeholder = st.empty()
        # CREATES: Empty container that we'll update
        # PURPOSE: Single element we can modify repeatedly
        
        full = ""  # Accumulator for complete response
        
        for chunk in response.response_gen:
            # ITERATOR: Generator that yields text chunks
            # YIELDS: Individual tokens or small phrases
            # RATE: ~50-100 tokens/second from Groq
            
            full += chunk  # Append to accumulator
            placeholder.markdown(full + "▌")  # Show with cursor
            # CURSOR: "▌" indicates typing in progress
            # UPDATE: Streamlit efficiently re-renders same element
        
        placeholder.markdown(full)  # Final text without cursor
        # CLEANUP: Remove typing indicator
        
        
        # ====================================================================
        # SOURCE CITATION
        # ====================================================================
        
        # Extract URLs from source nodes
        sources = [
            f"[{i+1}]({n.metadata.get('source_url','')})" 
            for i, n in enumerate(response.source_nodes[:7])  # Top 7 only
            if n.metadata.get("source_url")  # Skip if missing URL
        ]
        # FORMAT: [1](url) [2](url) [3](url)
        # CLICKABLE: Markdown links
        # LIMIT: 7 sources (keeps UI clean)
        
        if sources:
            # DISPLAY: Only if we have sources
            st.caption("Sources: " + " • ".join(sources))
            # CAPTION: Small text, de-emphasized
            # SEPARATOR: Bullet points for readability
        
        
        # ====================================================================
        # CONFIDENCE WARNING
        # ====================================================================
        
        if bundle.low_confidence:
            # THRESHOLD: avg_score < 0.80 (configurable)
            
            if bundle.fallback_results:
                # Case 1: Web search succeeded
                warning_msg = (
                    f"⚠️ Low confidence on internal docs (avg score {bundle.avg_score:.2f}). "
                    "Pulling in live context…"
                )
                # TRANSPARENT: Show exact score
                # EXPLANATION: Why we're using web search
            else:
                # Case 2: Web search disabled or failed
                warning_msg = (
                    "⚠️ Low confidence on internal docs. "
                    "You may need to re-ingest or broaden the crawl."
                )
                # ACTIONABLE: Suggests fix
            
            st.warning(warning_msg)
            # WARNING STYLE: Yellow box, attention-grabbing
        
        
        # ====================================================================
        # WEB FALLBACK DISPLAY
        # ====================================================================
        
        if bundle.fallback_answer:
            # PRESENT: Only if web search was triggered
            
            st.markdown("#### Live Web Update")
            # SECTION HEADER: Clearly separates web content
            
            st.markdown(bundle.fallback_answer)
            # CONTENT: LLM synthesis of web search results
            
            if bundle.fallback_results:
                # METADATA: Individual search results
                st.caption("Fresh sources")
                # LABEL: Distinguish from internal sources
                
                for item in bundle.fallback_results:
                    # ITERATE: Each web search result
                    title = item.get("title") or "Result"
                    url = item.get("url") or ""
                    summary = item.get("content") or ""
                    
                    # Truncate long summaries
                    summary_truncated = summary[:220]
                    ellipsis = '…' if len(summary) > 220 else ''
                    
                    st.markdown(f"- [{title}]({url}) — {summary_truncated}{ellipsis}")
                    # FORMAT: Bullet list of clickable links
                    # TRUNCATED: 220 chars (about 2 lines)
        
        
        # ====================================================================
        # EXPANDABLE SOURCE DETAILS
        # ====================================================================
        
        with st.expander(f"Sources ({len(response.source_nodes)} chunks)", expanded=False):
            # EXPANDER: Collapsed by default (keeps UI clean)
            # COUNT: Shows total chunks used
            
            for i, node in enumerate(response.source_nodes[:10]):  # Top 10 only
                # LIMIT: 10 chunks (full details would be overwhelming)
                
                score = node.score or 0.0  # Similarity score
                title = node.metadata.get("title", "Untitled")
                source_url = node.metadata.get("source_url")
                
                # Build clickable link if URL exists
                link_text = f"→ [{source_url}]({source_url})  " if source_url else ""
                
                # Chunk header with metadata
                st.markdown(
                    f"**[{i+1}]** {title}  {link_text}`relevance: {score:.3f}`"
                )
                # COMPONENTS:
                # - [1]: Number for reference
                # - Title: Document title
                # - →: Arrow indicating external link
                # - URL: Clickable source
                # - relevance: Similarity score (0.0-1.0)
                
                # Chunk content preview
                with st.container():
                    # CONTAINER: Visual grouping
                    snippet = node.get_text()[:800]  # First 800 chars
                    st.markdown(f"> {snippet}…")
                    # BLOCKQUOTE: "> " prefix for visual distinction
                    # TRUNCATED: Prevents UI overflow
        
        
        # ====================================================================
        # SAVE TO HISTORY
        # ====================================================================
        
        # Combine answer and web fallback (if present)
        assistant_content = full
        if bundle.fallback_answer:
            assistant_content = f"{full}\n\n---\n**Live web update**\n{bundle.fallback_answer}"
            # SEPARATOR: Horizontal rule (---) for visual break
    
    # Add assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_content,
        "sources": sources
    })
    # PERSISTS: For display on next rerun


# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    # SIDEBAR: Persistent info panel on left/right
    
    
    # ========================================================================
    # STORAGE STATUS INDICATOR
    # ========================================================================
    
    st.markdown("**Last updated**")
    # LABEL: When index was last modified
    
    storage_path = settings.storage_dir  # Path object
    
    if storage_path.exists():
        # Storage directory exists, check files
        files = [p for p in storage_path.glob("**/*") if p.is_file()]
        # RECURSIVE: Glob all files in subdirectories
        
        if files:
            # Files present, find most recent
            ts = max(p.stat().st_mtime for p in files)
            # TIMESTAMP: Unix timestamp of most recent modification
            
            human = datetime.fromtimestamp(ts).strftime("%b %d, %Y %H:%M")
            # FORMAT: "Nov 25, 2025 14:30"
            
            st.success(f"{human} UTC", icon="🟢")
            # GREEN INDICATOR: System is fresh
        else:
            # Directory exists but empty
            st.info("Indexed storage folder is empty", icon="⏳")
            # INFO: Not critical, but needs attention
    else:
        # Directory doesn't exist
        st.info("Not indexed yet", icon="⏳")
        # ACTIONABLE: User needs to run ingest.py
    
    
    # ========================================================================
    # SYSTEM CONFIGURATION DISPLAY
    # ========================================================================
    
    st.divider()  # Visual separator
    
    st.write(f"**Domain:** {settings.domain_url}")
    # SHOWS: What website is indexed
    # USEFUL: Multi-tenant deployments
    
    st.write(f"**Index:** `{settings.index_name}`")
    # SHOWS: Pinecone index name
    # FORMAT: Code style for technical accuracy
    
    
    # ========================================================================
    # DEPLOYMENT LINK
    # ========================================================================
    
    #st.divider()
    
    #st.markdown("Deploy free → [share.streamlit.io](https://share.streamlit.io)")
    # CTA: Encourages others to deploy
    # VALUE: Shows accessibility (free deployment)
    
    
    # ========================================================================
    # ABOUT THE AUTHOR
    # ========================================================================
    
    st.divider()
    
    st.markdown("### 👤 About")
    st.markdown("Built by **[Shawn Slattery](https://www.linkedin.com/in/shawn-slattery-843654201/)**")
    # ATTRIBUTION: Your name prominently displayed
    # CLICKABLE: Direct link to LinkedIn
    
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-shawnslat-181717?logo=github)](https://github.com/shawnslat)")
    # BADGE: Professional-looking GitHub link
    # VISUAL: Shield.io badge for polish
    
    
    # ========================================================================
    # ANALYTICS OPT-IN
    # ========================================================================
    
    st.divider()
    
    st.checkbox(
        "Allow anonymous analytics (Umami)",
        value=st.session_state.analytics_opt_in,
        key="analytics_opt_in",  # Binds to session state
        help="Optional: help us track page views. No PII.",
        # TOOLTIP: Appears on hover
    )
    # PRIVACY: Opt-in, not opt-out (GDPR compliant)
    # TRANSPARENCY: Explains what's tracked


# ============================================================================
# ANALYTICS INJECTION (CONDITIONAL)
# ============================================================================

if st.session_state.get("analytics_opt_in"):
    # ONLY IF: User explicitly opted in
    
    st.markdown(
        """
        <script async defer data-website-id="umami-placeholder" src="https://analytics.example.com/script.js"></script>
        """,
        unsafe_allow_html=True,
    )
    # UMAMI: Privacy-focused analytics (GDPR compliant)
    # ALTERNATIVE: Google Analytics, Plausible, Fathom
    # TODO: Replace placeholder with real Umami site ID


# ============================================================================
# STREAMLIT-SPECIFIC NOTES
# ============================================================================

"""
STREAMLIT EXECUTION MODEL:
--------------------------
Streamlit reruns entire script on every user interaction:
- Button click → Full rerun
- Text input → Full rerun  
- Slider change → Full rerun

Optimizations:
1. Session state for persistence
2. @st.cache_data for expensive computations
3. st.empty() for efficient updates

DEPLOYMENT CHECKLIST:
---------------------
1. Test locally: streamlit run app.py
2. Push to GitHub (public or private)
3. Connect at share.streamlit.io
4. Add secrets in dashboard:
   - GROQ_API_KEY
   - PINECONE_API_KEY
   - FIRECRAWL_API_KEY
   - NOMIC_API_KEY
5. Deploy! (auto-updates on git push)

COMMON ISSUES:
--------------
1. "Module not found"
   → Add to requirements.txt
   
2. "Session state lost"
   → Check for st.rerun() or form submissions
   
3. "Slow performance"
   → Use st.cache_data for model loading
   
4. "Rate limit errors"
   → Increase cooldown in rate limiting code
   
5. "Memory errors"
   → Reduce batch sizes, use smaller models

CUSTOMIZATION IDEAS:
--------------------
1. Add file upload (PDF, DOCX)
2. Add voice input (speech-to-text)
3. Add export to PDF
4. Add conversation export
5. Add multi-language support
6. Add dark mode toggle
7. Add custom CSS themes
8. Add A/B testing for prompts
"""


# ============================================================================
# PERFORMANCE MONITORING (OPTIONAL)
# ============================================================================

"""
Add this after query_engine() call to track performance:

import time
start = time.time()
bundle = query_engine(prompt)
latency_ms = (time.time() - start) * 1000

# Log to monitoring service
logger.info(
    "Query completed",
    extra={
        "latency_ms": latency_ms,
        "avg_score": bundle.avg_score,
        "low_confidence": bundle.low_confidence,
        "source_count": len(response.source_nodes),
    }
)

# Display to user (dev mode only)
if os.getenv("DEBUG"):
    st.caption(f"⚡ {latency_ms:.0f}ms | Confidence: {bundle.avg_score:.2f}")
"""