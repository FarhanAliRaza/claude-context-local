#!/usr/bin/env python3
"""Quick test to verify llama-server embedder works."""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from embeddings.llama_embedder import LlamaServerEmbedder

def test_llama_embedder():
    """Test basic llama-server embedder functionality."""
    print("Testing llama-server embedder...")

    # Initialize embedder
    embedder = LlamaServerEmbedder(
        base_url="http://localhost:10101",
        model_name="nomic-embed-text"
    )

    # Test single query embedding
    print("\n1. Testing single query embedding...")
    query = "def calculate_sum(a, b):\n    return a + b"
    embedding = embedder.embed_query(query)
    print(f"   ✓ Generated embedding with shape: {embedding.shape}")
    print(f"   ✓ Embedding dimension: {len(embedding)}")

    # Test batch embeddings
    print("\n2. Testing batch embeddings...")
    queries = [
        "def add(x, y): return x + y",
        "class User:\n    def __init__(self, name):\n        self.name = name",
        "import numpy as np"
    ]
    embeddings = embedder._get_embeddings(queries)
    print(f"   ✓ Generated {len(embeddings)} embeddings")
    print(f"   ✓ Each embedding dimension: {len(embeddings[0])}")

    # Test model info
    print("\n3. Testing model info...")
    info = embedder.get_model_info()
    print(f"   ✓ Model: {info['model_name']}")
    print(f"   ✓ URL: {info['base_url']}")
    print(f"   ✓ Dimension: {info['embedding_dimension']}")
    print(f"   ✓ Status: {info['status']}")

    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_llama_embedder()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
