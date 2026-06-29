import os
import hashlib
import sqlite3
import json
import math
import ollama

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes_vector.db")
EMBED_MODEL = "qwen3-embedding:4b"

class VectorDBManager:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        # Ensure database directory exists
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        
        # Initialize SQLite database for local vector storage (removes unstable ChromaDB binary dependency)
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        """Create local chunks table if it does not exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT,
                chunk_index INTEGER,
                document_name TEXT,
                text TEXT,
                embedding TEXT  -- JSON serialized list of floats
            )
        """)
        self.conn.commit()

    def get_md5_hash(self, text: str) -> str:
        """Generate MD5 hash of text for unique chunk ID."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embedding(self, text: str):
        """Fetch embedding from local Ollama service."""
        try:
            response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error with {EMBED_MODEL}, trying fallback: {e}")
            try:
                response = ollama.embeddings(model="qwen3:4b", prompt=text)
                return response["embedding"]
            except Exception as e2:
                raise RuntimeError(f"Ollama embedding failure: {e2}")

    def add_chunk(self, doc_id: str, chunk_index: int, text: str, doc_name: str) -> str:
        """
        Add or update (upsert) a text chunk into SQLite.
        Uses MD5 hash of text to ensure deduplication.
        Returns the unique chunk ID.
        """
        chunk_hash = self.get_md5_hash(text)
        chunk_id = f"chk_{chunk_hash}"
        
        # Calculate embedding
        embedding = self.get_embedding(text)
        embedding_json = json.dumps(embedding)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO chunks (id, document_id, chunk_index, document_name, text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                document_id=excluded.document_id,
                chunk_index=excluded.chunk_index,
                document_name=excluded.document_name,
                text=excluded.text,
                embedding=excluded.embedding
        """, (chunk_id, doc_id, chunk_index, doc_name, text, embedding_json))
        self.conn.commit()
        return chunk_id

    def cosine_similarity(self, v1: list, v2: list) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(b * b for b in v2))
        if norm_v1 == 0.0 or norm_v2 == 0.0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def query_vector(self, query_text: str, n_results: int = 5):
        """Query SQLite database and rank results using cosine similarity."""
        query_embedding = self.get_embedding(query_text)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, document_id, chunk_index, document_name, text, embedding FROM chunks")
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            chunk_id, doc_id, chunk_idx, doc_name, text, embedding_json = row
            try:
                embedding = json.loads(embedding_json)
                similarity = self.cosine_similarity(query_embedding, embedding)
                
                results.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        "document_id": doc_id,
                        "chunk_index": chunk_idx,
                        "document_name": doc_name,
                        "length": len(text)
                    },
                    "similarity": similarity
                })
            except Exception as e:
                print(f"Error parsing embedding for chunk {chunk_id}: {e}")
                
        # Sort by similarity descending
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Take top N
        top_results = results[:n_results]
        
        # Map similarity to distance (for interface compatibility if needed)
        for r in top_results:
            r["distance"] = 1.0 - r["similarity"]
            
        return top_results

    def get_chunks_by_ids(self, chunk_ids: list):
        """Retrieve specific chunks directly by their IDs."""
        if not chunk_ids:
            return []
        
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT id, document_id, chunk_index, document_name, text 
            FROM chunks 
            WHERE id IN ({placeholders})
        """, chunk_ids)
        rows = cursor.fetchall()
        
        retrieved_chunks = []
        for row in rows:
            chunk_id, doc_id, chunk_idx, doc_name, text = row
            retrieved_chunks.append({
                "id": chunk_id,
                "text": text,
                "metadata": {
                    "document_id": doc_id,
                    "chunk_index": chunk_idx,
                    "document_name": doc_name
                }
            })
        return retrieved_chunks

    def delete_document_chunks(self, doc_id: str):
        """Delete all chunks belonging to a document ID."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self.conn.commit()

    def reset_db(self):
        """Reset the SQLite vector table."""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS chunks")
        self.conn.commit()
        self.create_tables()

    def get_stats(self):
        """Get database stats (total chunks count)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT count(*) FROM chunks")
        count = cursor.fetchone()[0]
        return {
            "total_chunks": count
        }
