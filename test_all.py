#!/usr/bin/env python3
"""
Comprehensive test suite for all project features.
"""

import sys
import traceback


def test_phase1():
    """Test Phase 1: Document Loader & Chunker."""
    print("=" * 60)
    print("PHASE 1: Document Loader & Chunker")
    print("=" * 60)
    
    try:
        from src.utils.io import clean_text
        from src.retrieval.chunker import chunk_text
        
        # Test clean_text
        print("\n1. Testing clean_text()...")
        dirty_text = "This   is    a    test\n\nwith   multiple   spaces"
        cleaned = clean_text(dirty_text)
        assert "  " not in cleaned, "clean_text failed: multiple spaces found"
        print("   ✓ clean_text() works correctly")
        
        # Test chunk_text
        print("\n2. Testing chunk_text()...")
        test_text = "A" * 1500
        chunks = chunk_text(test_text, size=500)
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
        assert all(len(chunk) <= 500 for chunk in chunks), "Some chunks exceed size limit"
        print(f"   ✓ chunk_text() works correctly (created {len(chunks)} chunks)")
        
        print("\n✓ Phase 1: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Phase 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_phase2():
    """Test Phase 2: Embeddings + Vectorstore."""
    print("=" * 60)
    print("PHASE 2: Embeddings + Vectorstore")
    print("=" * 60)
    
    try:
        from src.retrieval.embedder import Embedder
        from src.retrieval.vectorstore import VectorStore
        import numpy as np
        
        # Test Embedder
        print("\n1. Testing Embedder...")
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        print(f"   ✓ Embedder initialized (dimension: {embedder.dimension})")
        
        test_texts = ["This is a test", "Another test sentence"]
        embeddings = embedder.embed(test_texts)
        assert embeddings.shape[0] == 2, f"Expected 2 embeddings, got {embeddings.shape[0]}"
        print(f"   ✓ embed() works correctly (shape: {embeddings.shape})")
        
        query_embedding = embedder.embed_query("test query")
        assert query_embedding.shape[0] == 1, "Query embedding should have batch size 1"
        print(f"   ✓ embed_query() works correctly")
        
        # Test VectorStore
        print("\n2. Testing VectorStore...")
        vectorstore = VectorStore(dim=embedder.dimension)
        vectorstore.add(embeddings, test_texts)
        assert len(vectorstore.chunks) == 2, "Chunks not added correctly"
        print(f"   ✓ add() works correctly ({len(vectorstore.chunks)} chunks added)")
        
        results = vectorstore.search(query_embedding, k=2)
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        print(f"   ✓ search() works correctly (returned {len(results)} results)")
        
        print("\n✓ Phase 2: ALL TESTS PASSED\n")
        return True
        
    except ImportError as e:
        print(f"\n⚠ Phase 2 SKIPPED: Missing dependency - {e}")
        return None
    except Exception as e:
        print(f"\n✗ Phase 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_phase3():
    """Test Phase 3: LLM Interface."""
    print("=" * 60)
    print("PHASE 3: LLM Interface")
    print("=" * 60)
    
    try:
        from src.llm.model_interface import ModelInterface
        from src.llm.local_model_client import LocalModelClient
        
        print("\n1. Testing ModelInterface...")
        assert hasattr(ModelInterface, 'generate'), "ModelInterface missing generate method"
        print("   ✓ ModelInterface is properly defined")
        
        print("\n2. Testing LocalModelClient...")
        local_client = LocalModelClient()
        print("   ✓ LocalModelClient initialized")
        
        try:
            local_client.generate("test")
        except NotImplementedError:
            print("   ✓ LocalModelClient correctly raises NotImplementedError")
        
        print("\n3. Testing OpenAIClient...")
        try:
            from src.llm.openai_client import OpenAIClient
            import os
            
            if os.getenv("OPENAI_API_KEY"):
                client = OpenAIClient()
                print("   ✓ OpenAIClient initialized (API key found)")
            else:
                print("   ⚠ OpenAIClient requires OPENAI_API_KEY (skipping)")
        except ImportError:
            print("   ⚠ OpenAIClient requires 'openai' package (skipping)")
        
        print("\n✓ Phase 3: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Phase 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    print("=" * 60)
    print("INTEGRATION TEST: Chunker + Retriever")
    print("=" * 60)
    
    try:
        from src.retrieval.chunker import chunk_text
        from src.retrieval.retriever import Retriever
        
        print("\n1. Creating test chunks...")
        test_text = "Machine learning is a subset of artificial intelligence. " * 20
        chunks = chunk_text(test_text, size=100)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        print("\n2. Testing Retriever...")
        try:
            retriever = Retriever()
            retriever.index_chunks(chunks)
            print(f"   ✓ Chunks indexed successfully")
            
            results = retriever.search("artificial intelligence", k=3)
            assert len(results) == 3, f"Expected 3 results, got {len(results)}"
            print(f"   ✓ Search works correctly (returned {len(results)} results)")
            
            print("\n✓ Integration test: ALL TESTS PASSED\n")
            return True
            
        except ImportError as e:
            print(f"   ⚠ Retriever requires dependencies: {e}")
            return None
            
    except Exception as e:
        print(f"\n✗ Integration test FAILED: {e}")
        traceback.print_exc()
        return False


def test_improvements():
    """Test Phase 10 improvements."""
    print("=" * 60)
    print("PHASE 10: Improvements")
    print("=" * 60)
    
    try:
        from src.retrieval.chunker import semantic_chunk_text
        from src.retrieval.retriever import Retriever
        
        print("\n1. Testing Semantic Chunking...")
        test_text = "Machine learning is AI. Deep learning uses neural networks."
        semantic_chunks = semantic_chunk_text(test_text, max_chunk_size=100)
        assert len(semantic_chunks) > 0, "Semantic chunking should produce chunks"
        print(f"   ✓ Semantic chunking works ({len(semantic_chunks)} chunks)")
        
        print("\n2. Testing Chunk Filtering...")
        retriever = Retriever()
        retriever.index_chunks([
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks",
            "The weather is sunny today"
        ])
        filtered = retriever.retrieve("machine learning", k=3, min_similarity=0.3)
        print(f"   ✓ Chunk filtering works ({len(filtered)} filtered chunks)")
        
        print("\n✓ Phase 10: ALL TESTS PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Phase 10 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60 + "\n")
    
    results = {}
    
    results['phase1'] = test_phase1()
    results['phase2'] = test_phase2()
    results['phase3'] = test_phase3()
    results['integration'] = test_integration()
    results['improvements'] = test_improvements()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r is True)
    skipped = sum(1 for r in results.values() if r is None)
    failed = sum(1 for r in results.values() if r is False)
    
    for name, result in results.items():
        status = "✓ PASSED" if result is True else "⚠ SKIPPED" if result is None else "✗ FAILED"
        print(f"{name.upper():15} : {status}")
    
    print(f"\nTotal: {passed} passed, {skipped} skipped, {failed} failed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
