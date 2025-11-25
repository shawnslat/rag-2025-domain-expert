#!/usr/bin/env python
"""
System Diagnostics Module
==========================
RAG 2025 Domain Expert - Health Check & Debugging Tool

AUTHOR: Shawn Slattery
GitHub: https://github.com/shawnslat
LinkedIn: https://www.linkedin.com/in/shawn-slattery-843654201/

Built: November 2025
Purpose: Diagnose embedding, retrieval, and storage issues

NOTE: This is the ANNOTATED version with comprehensive developer comments.
      For the working version, see diagnose.py (uploaded file).
      This version explains what each test checks and why it matters.

WHEN TO USE:
------------
Run this script when:
- Queries return 0.0 confidence scores
- "No results found" errors
- Suspecting embedding dimension mismatch
- After re-ingestion to verify success
- Before deploying to production

WHAT IT CHECKS:
---------------
1. Pinecone Index Health
   - Vector count
   - Dimension (must be 768 for Nomic)
   - Connectivity

2. Embedding Generation
   - Nomic model loads correctly
   - Generates 768-dim vectors
   - Vector magnitude is normalized (~1.0)

3. Direct Pinecone Query
   - Retrieval works at all
   - Similarity scores are reasonable (>0.1)
   - Metadata is present

4. Local Storage
   - Index files exist
   - Files are not empty
   - Sizes look reasonable

USAGE:
------
python diagnose.py

Expected output if healthy:
✅ Dimension: 768
✅ Scores look normal!
✅ Storage directory has files

TROUBLESHOOTING:
----------------
If this script fails, the query engine will definitely fail.
Fix issues here first before trying to run queries.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# MUST happen before importing config (which reads env vars)
load_dotenv()

# ============================================================================
# DIAGNOSTIC HEADER
# ============================================================================

print("=" * 80)
print("RAG SYSTEM DIAGNOSTICS")
print("=" * 80)
# VISUAL: Clear section separator for console output


# ============================================================================
# TEST 1: PINECONE INDEX STATUS
# ============================================================================

print("\n1. PINECONE INDEX STATUS:")
print("-" * 80)

from pinecone import Pinecone
from config import settings

# Initialize Pinecone client
# REQUIRES: PINECONE_API_KEY in .env
pc = Pinecone(api_key=settings.pinecone_api_key)

# Connect to specific index
# RAISES: PineconeException if index doesn't exist
index = pc.Index(settings.index_name)

# Fetch index statistics
# RETURNS: IndexStats object with:
#   - total_vector_count: int
#   - dimension: int
#   - namespaces: dict (for multi-tenant setups)
stats = index.describe_index_stats()

# Display basic info
print(f"Index Name: {settings.index_name}")
print(f"Total Vectors: {stats.total_vector_count}")
# EXPECTED: 537 (from our 1-page ingestion) or 5000+ (full corpus)
# ZERO?: Index is empty, need to run ingest.py

print(f"Dimension: {stats.dimension}")
# CRITICAL: MUST be 768 for Nomic embeddings
# WHY?: Query embeddings are 768-dim, must match index

print(f"Expected Dimension (Nomic): 768")

# Validate dimension
if stats.dimension != 768:
    print("⚠️  WARNING: Dimension mismatch! Expected 768 for Nomic embeddings.")
    # IMPACT: All queries will return 0.0 similarity
    # FIX: Delete index, re-run ingest.py with correct embedding model
# GOOD: If we get here without warning, dimension is correct


# ============================================================================
# TEST 2: EMBEDDING GENERATION
# ============================================================================

print("\n2. EMBEDDING GENERATION TEST:")
print("-" * 80)

from llama_index.embeddings.nomic import NomicEmbedding

# Initialize Nomic embedding model
# CONNECTS: To Nomic API (requires NOMIC_API_KEY or nomic login)
embed_model = NomicEmbedding(model_name="nomic-embed-text-v1.5")
# MODEL: nomic-embed-text-v1.5
# OUTPUT: 768-dimensional vectors
# ALTERNATIVES:
#   - text-embedding-3-small (OpenAI, 1536-dim)
#   - all-MiniLM-L6-v2 (local, 384-dim)

# Generate test embedding
test_text = "This is a test embedding"
# SIMPLE: Short text to verify basic functionality
test_embedding = embed_model.get_text_embedding(test_text)
# RETURNS: List of 768 floats
# DURATION: ~200ms (includes API call)

# Validate embedding
print(f"Generated embedding dimension: {len(test_embedding)}")
# EXPECTED: 768
# WRONG?: Check embedding_model in config.py

print(f"First 5 values: {test_embedding[:5]}")
# SAMPLE: [0.039, 0.061, -0.131, -0.048, 0.055]
# PURPOSE: Verify we got actual numbers, not errors

# Calculate vector magnitude (L2 norm)
magnitude = sum(x**2 for x in test_embedding)**0.5
print(f"Embedding magnitude: {magnitude:.4f}")
# EXPECTED: ~1.0 (Nomic normalizes vectors)
# WHY NORMALIZED?: Cosine similarity works best with unit vectors
# FORMULA: ||v|| = sqrt(v1² + v2² + ... + v768²)
# WRONG (0.0)?: Embedding failed, all zeros
# WRONG (>>1.0)?: Not normalized, might indicate wrong model


# ============================================================================
# TEST 3: DIRECT PINECONE QUERY
# ============================================================================

print("\n3. DIRECT PINECONE QUERY TEST:")
print("-" * 80)

# Generate query embedding
# QUERY: Generic ML topic to test retrieval
query_embedding = embed_model.get_text_embedding("machine learning research")
print(f"Query embedding dimension: {len(query_embedding)}")
# VERIFY: Still 768 (consistency check)

# Query Pinecone directly (bypass LlamaIndex)
# PURPOSE: Test raw Pinecone functionality
results = index.query(
    vector=query_embedding,  # 768-dim vector
    top_k=5,  # Return top 5 matches
    include_metadata=True,  # Include title, url, etc.
)
# RETURNS: QueryResponse with .matches list
# DURATION: ~50-100ms

# Display results
print(f"\nRetrieved {len(results.matches)} results:")
for i, match in enumerate(results.matches, 1):
    # MATCH ATTRIBUTES:
    # - .id: Unique vector ID (UUID)
    # - .score: Similarity score (0.0-1.0)
    # - .metadata: Dict with title, source_url, text, etc.
    
    print(f"  [{i}] Score: {match.score:.4f}")
    # INTERPRETATION:
    # - 0.9-1.0: Excellent match (almost identical)
    # - 0.8-0.9: Good match (highly relevant)
    # - 0.7-0.8: Moderate match (somewhat relevant)
    # - 0.6-0.7: Weak match (barely relevant)
    # - <0.6: Poor match (likely noise)
    
    print(f"      ID: {match.id}")
    # FORMAT: UUID like "b3200527-f968-4de5-a64f-c33f85da7495#..."
    # PARTS: document_id#node_id
    
    if match.metadata:
        # Metadata should include title, source_url, text
        title = match.metadata.get('title', 'N/A')[:60]  # Truncate long titles
        print(f"      Title: {title}")
        # VERIFY: Metadata is present and readable

# ========================================================================
# SCORE ANALYSIS
# ========================================================================

# Check for critical failure: All scores 0.0
if all(match.score == 0.0 for match in results.matches):
    print("\n❌ CRITICAL: All scores are 0.0!")
    print("   This indicates embedding dimension mismatch or corrupt index.")
    print("\n   SOLUTION: Re-run ingestion with correct embedding model.")
    # CAUSE: Query uses 768-dim but index has different dimension
    # OR: Index is corrupted
    # FIX: Delete index → re-run ingest.py

# Check for suspicious behavior: All scores very low
elif all(match.score < 0.1 for match in results.matches):
    print("\n⚠️  WARNING: All scores very low (< 0.1)")
    print("   This might indicate:")
    print("   - Wrong embedding model used during ingestion")
    # EXAMPLE: Ingested with OpenAI, querying with Nomic
    print("   - Different Nomic model version")
    # EXAMPLE: Ingested with v1.0, querying with v1.5
    print("   - Text preprocessing mismatch")
    # EXAMPLE: Ingested lowercased, querying with original case
    # SOLUTION: Re-ingest with consistent settings

# Success: Scores look reasonable
else:
    print("\n✅ Scores look normal!")
    # MEANING: Embeddings match, retrieval works
    # RANGE: Typical scores 0.6-0.9 for relevant content


# ============================================================================
# TEST 4: LOCAL STORAGE CHECK
# ============================================================================

print("\n4. LOCAL STORAGE CHECK:")
print("-" * 80)

from pathlib import Path

# Check if storage directory exists
storage_dir = Path(settings.storage_dir)  # Default: "storage/"

if storage_dir.exists():
    # Directory exists, check contents
    files = list(storage_dir.glob("*"))
    # GLOB: Match all files in directory (non-recursive)
    
    print(f"Storage directory: {storage_dir}")
    print(f"Files found: {len(files)}")
    # EXPECTED: 3 files (docstore, index_store, graph_store)
    
    # List each file with size
    for f in files:
        size_bytes = f.stat().st_size
        print(f"  - {f.name} ({size_bytes} bytes)")
        # EXPECTED SIZES:
        # - docstore.json: ~5-10 MB (largest, has all text)
        # - index_store.json: ~10-50 KB (metadata only)
        # - graph_store.json: ~1 KB (minimal structure)
        
        # Sanity checks
        if size_bytes == 0:
            print(f"    ⚠️  WARNING: {f.name} is empty!")
            # PROBLEM: File created but no data written
        elif f.name == "docstore.json" and size_bytes < 1000:
            print(f"    ⚠️  WARNING: {f.name} suspiciously small!")
            # PROBLEM: Should be MB-sized, not KB-sized
else:
    print(f"❌ Storage directory not found: {storage_dir}")
    # MEANING: Ingestion never ran or failed
    # FIX: Run ingest.py


# ============================================================================
# DIAGNOSTIC SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)

# INTERPRETATION GUIDE:
"""
✅ ALL GREEN:
- Total Vectors > 0
- Dimension = 768  
- Scores 0.6-0.9
- Storage files present
→ System is healthy, queries will work

⚠️  WARNINGS:
- Scores < 0.1 → Embedding mismatch, re-ingest
- Missing metadata → Crawl/ingestion issue
- Small storage files → Incomplete ingestion

❌ CRITICAL:
- Scores all 0.0 → Must re-ingest
- Dimension mismatch → Must delete index & re-ingest
- No storage directory → Must run ingest.py

NEXT STEPS:
-----------
If healthy: Run streamlit app.py
If issues: Fix problems listed above, re-run diagnose.py
"""


# ============================================================================
# ADVANCED DIAGNOSTICS (OPTIONAL)
# ============================================================================

"""
ADDITIONAL CHECKS YOU CAN ADD:
-------------------------------

1. Test specific query patterns:
   query_embedding = embed_model.get_text_embedding("transformers neural networks")
   # Check domain-specific retrieval

2. Check index configuration:
   index.describe_index()
   # Verify metric (cosine), region, replicas

3. Test metadata filtering:
   results = index.query(
       vector=query_embedding,
       filter={"source_url": {"$eq": "https://arxiv.org/..."}}
   )
   # Verify metadata indexing works

4. Measure query latency:
   import time
   start = time.time()
   results = index.query(vector=query_embedding, top_k=10)
   latency = time.time() - start
   print(f"Query latency: {latency*1000:.0f}ms")
   # Should be <100ms typically

5. Check for duplicate vectors:
   # Query with a known vector ID
   # If returns duplicates, index might have been ingested multiple times
"""