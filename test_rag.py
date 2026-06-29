import os
import sys
from Veritabanı.vector_db import VectorDBManager
from Veritabanı.graph_db import GraphDBManager, Neo4jConnectionError
from Modeller.extractor import DocumentExtractor
from Modeller.retriever import HybridRetriever
from Modeller.synthesizer import AnswerSynthesizer

def test_pipeline():
    print("=== HERMES PIPELINE TESTING ===")
    
    # 1. Test Database Configurations
    print("\n[Step 1] Initializing Vector DB (Chroma)...")
    try:
        vdb = VectorDBManager()
        print("Vector DB initialized successfully. Total chunks currently indexed:", vdb.get_stats()["total_chunks"])
    except Exception as e:
        print(f"FAILED to initialize Vector DB: {e}")
        return False

    print("\n[Step 2] Connecting to Graph DB (Neo4j)...")
    try:
        gdb = GraphDBManager(uri="bolt://localhost:7687", user="neo4j", password="hermes123")
        print("Graph DB connected successfully. Current stats:", gdb.get_stats())
    except Neo4jConnectionError as e:
        print(f"FAILED to connect to Neo4j database: {e}")
        print("\nNOTE: Please make sure your Neo4j Server is running on bolt://localhost:7687")
        return False
    except Exception as e:
        print(f"Unexpected Graph DB connection failure: {e}")
        return False

    # Clean test elements
    print("\n[Step 3] Adding test concept nodes & relations directly...")
    try:
        doc_id = "doc_test_123"
        doc_name = "test_document.pdf"
        chunk_id = "chk_test_abc"
        
        # Add document itself
        gdb.add_concept_node(name=doc_name, description="Test PDF file node", doc_id=doc_id, chunk_id="doc_node")
        # Add concept A
        gdb.add_concept_node(name="Agentic Coding", description="Autonomous software creation by AI agents", doc_id=doc_id, chunk_id=chunk_id)
        # Add concept B
        gdb.add_concept_node(name="Antigravity", description="Gemini-based AI pair programming tool", doc_id=doc_id, chunk_id=chunk_id)
        # Link document to concepts
        gdb.add_relationship(source_name=doc_name, target_name="Agentic Coding", rel_type="DISCUSSES", description="Discusses agentic coding paradigms")
        gdb.add_relationship(source_name=doc_name, target_name="Antigravity", rel_type="DISCUSSES", description="Mentions Antigravity tool")
        # Add relation between A and B
        gdb.add_relationship(source_name="Antigravity", target_name="Agentic Coding", rel_type="ENABLES", description="Antigravity enables agentic coding workflows")
        
        print("Test nodes and relationships added to Neo4j successfully.")
    except Exception as e:
        print(f"FAILED to write to Graph DB: {e}")
        gdb.close()
        return False

    print("\n[Step 4] Adding test chunk to Vector DB...")
    try:
        test_text = "Antigravity is a powerful AI coding assistant created by Google DeepMind. It enables advanced Agentic Coding workflows directly on the user's system."
        vdb.add_chunk(doc_id=doc_id, chunk_index=0, text=test_text, doc_name=doc_name)
        print("Test text chunk added to ChromaDB successfully.")
    except Exception as e:
        print(f"FAILED to write to Vector DB: {e}")
        gdb.close()
        return False

    # 2. Test Retrieval logic
    print("\n[Step 5] Testing Hybrid RAG Retrieval...")
    try:
        retriever = HybridRetriever(vdb, gdb)
        query = "Antigravity and Agentic Coding"
        print(f"Querying: '{query}'")
        retrieved_data = retriever.retrieve(query)
        
        print(f"- Routing decision: {retrieved_data['routing']}")
        print(f"- Sources found: {retrieved_data['sources']}")
        print(f"- Context Text Length: {len(retrieved_data['context_text'])} chars")
        print(f"- Relationships found: {len(retrieved_data['relationships'])}")
        
        if not retrieved_data["sources"]:
            print("WARNING: Hybrid retriever returned no sources.")
    except Exception as e:
        print(f"FAILED hybrid retrieval: {e}")
        gdb.close()
        return False

    # 3. Test Synthesizer
    print("\n[Step 6] Testing LLM Answer Synthesis...")
    try:
        synthesizer = AnswerSynthesizer()
        print("Calling synthesizer LLM...")
        synth_result = synthesizer.synthesize(query, retrieved_data["context_text"])
        
        print("\n--- THOUGHT PROCESS ---")
        print(synth_result["thought"])
        print("\n--- SYNTHESIZED ANSWER ---")
        print(synth_result["answer"])
        print("--------------------------")
        
        if not synth_result["answer"]:
            print("WARNING: Synthesizer returned empty answer.")
    except Exception as e:
        print(f"FAILED answer synthesis: {e}")
        gdb.close()
        return False

    # Cleanup test data
    print("\n[Step 7] Cleaning up test data...")
    try:
        vdb.delete_document_chunks(doc_id)
        with gdb.driver.session() as session:
            session.run("MATCH (n:Concept) WHERE n.doc_id = $doc_id OR n.id = $doc_id DETACH DELETE n", doc_id=doc_id)
        print("Test data cleaned up successfully.")
    except Exception as e:
        print(f"Cleanup failed: {e}")

    gdb.close()
    print("\n=== PIPELINE TEST COMPLETED SUCCESSFULLY! ===")
    return True

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
