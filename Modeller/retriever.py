import re
import json
import ollama
from Veritabanı.vector_db import VectorDBManager
from Veritabanı.graph_db import GraphDBManager

class HybridRetriever:
    def __init__(self, vector_db: VectorDBManager, graph_db: GraphDBManager, model_name: str = "qwen3:4b"):
        self.vdb = vector_db
        self.gdb = graph_db
        self.model_name = model_name

    def extract_query_keywords(self, query: str, chat_history: list = None) -> list:
        """Extract key search terms from the query using a fast LLM call, resolving pronouns using history."""
        system_prompt = """
        IDENTITY:
        You are a search query keyword extractor.
        
        PURPOSE:
        Extract 1 to 4 core entities, nouns, or technical concepts from the query to be used in database lookup.
        If the query contains pronouns (like 'bunu', 'ona', 'it', 'they', 'onun') referring to concepts in the conversation history, resolve them and extract the original concepts.
        
        RULES:
        1. Output ONLY a valid JSON list of strings. No explanations, no markdown blocks.
        2. Extract concepts exactly as they appear or normalize them slightly (e.g., 'DEUS AI' -> 'DEUS AI').
        3. Do not include question words (who, what, how) or common verbs.
        
        EXAMPLES:
        Conversation History:
        User: "Oktavis nedir?"
        Assistant: "Oktavis, Meow entegrasyonu kullanan bir lojistik projesidir."
        Query: "Peki bunu kim kurdu?" -> ["Oktavis", "kurucu"]
        """
        
        history_str = ""
        if chat_history:
            history_str = "CONVERSATION HISTORY:\n"
            for turn in chat_history[-3:]: # Look at the last 3 turns
                role_name = "User" if turn["role"] == "user" else "Assistant"
                history_str += f"{role_name}: {turn['content']}\n"
            history_str += "\n"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{history_str}QUERY: {query}"}
                ],
                format="json",
                options={"temperature": 0.0, "num_predict": 128}
            )
            content = response["message"]["content"].strip()
            
            # Clean JSON parser: find first [...] in case a lightweight model outputs reasoning text
            json_match = re.search(r'\[\s*".*?"\s*(?:,\s*".*?"\s*)*\]', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                
            keywords = json.loads(content)
            if isinstance(keywords, list):
                return [k for k in keywords if isinstance(k, str) and k.strip()]
        except Exception:
            pass
            
        # Regex fallback: split by non-alphanumeric and filter short words
        words = re.findall(r'\b\w{3,}\b', query)
        # Remove common Turkish/English stopwords
        stopwords = {
            "nasıl", "nedir", "neler", "niçin", "neden", "hakkında", "ilişkin", "yaz", "listele", 
            "özetle", "açıkla", "göster", "ver", "what", "how", "why", "who", "show", "describe",
            "summarize", "about", "relation", "between"
        }
        keywords = [w for w in words if w.lower() not in stopwords]
        return keywords[:4]

    def is_summary_query(self, query: str) -> bool:
        """Determine if the query is a general summary or system overview request."""
        summary_keywords = [
            "özet", "neler var", "genel", "sistemde ne", "neler kayıtlı", "özetle", "dokümanlar", 
            "summary", "what is in", "overview", "list all", "database contain", "tümünü"
        ]
        query_lower = query.lower()
        return any(k in query_lower for k in summary_keywords)

    def retrieve(self, query: str, top_k_vector: int = 4, top_k_graph: int = 4, chat_history: list = None) -> dict:
        """
        Runs the Two-Pass hybrid retrieval:
        1. Checks Router: If it's a summary or latest query, handles routing.
        2. Pass 1 (Graph): Extract query keywords, find matching concepts, retrieve their chunk IDs.
        3. Pass 2 (Vector): Fetch chunks corresponding to those chunk IDs.
        4. Pass 3 (Vector Semantic Search): Query ChromaDB for top_k_vector semantic matches.
        5. Merge and prioritize chunks.
        6. Gather relationship details from the Graph DB to append as structural context.
        """
        # Check if query is an overview/summary query
        if self.is_summary_query(query):
            graph_summary = self.gdb.get_system_summary()
            return {
                "routing": "summary",
                "context_text": graph_summary["summary"],
                "chunks": [],
                "relationships": [],
                "concepts_found": [],
                "sources": ["System Graph Metadata"]
            }

        # 2. KEYWORD EXTRACTION (with history context)
        keywords = self.extract_query_keywords(query, chat_history)
        print(f"Extracted search keywords: {keywords}")

        # 3. PASS 1: Search Graph DB for matching concepts
        graph_concepts = self.gdb.search_nodes_by_keywords(keywords)
        
        # Collect concept, document, project, and user IDs from graph search results
        concept_ids = []
        concepts_metadata = []
        matched_doc_ids = []
        matched_project_ids = []
        matched_user_ids = []
        
        for concept in graph_concepts:
            label = concept.get("label", "Concept")
            c_id = concept["id"]
            if label in ["Concept", "SystemMetadata"]:
                concept_ids.append(c_id)
                concepts_metadata.append(f"- {concept['name']}: {concept['description']}")
            elif label == "Document":
                matched_doc_ids.append(c_id)
            elif label == "Project":
                matched_project_ids.append(c_id)
            elif label == "User":
                matched_user_ids.append(c_id)

        # 1. Resolve Document IDs from Projects and Users via Graph connections
        proj_doc_ids = []
        if matched_project_ids:
            query_proj = "MATCH (doc:Document)-[:BELONGS_TO]->(p:Project) WHERE p.id IN $p_ids RETURN doc.id AS doc_id"
            try:
                with self.gdb.driver.session() as session:
                    res = session.run(query_proj, p_ids=matched_project_ids)
                    proj_doc_ids = [rec["doc_id"] for rec in res]
            except Exception:
                pass

        user_doc_ids = []
        if matched_user_ids:
            query_user = "MATCH (u:User)-[:UPLOADED]->(doc:Document) WHERE u.id IN $u_ids RETURN doc.id AS doc_id"
            try:
                with self.gdb.driver.session() as session:
                    res = session.run(query_user, u_ids=matched_user_ids)
                    user_doc_ids = [rec["doc_id"] for rec in res]
            except Exception:
                pass

        all_doc_ids = list(set(matched_doc_ids + proj_doc_ids + user_doc_ids))

        # 2. Get chunk IDs for matched documents from SQLite
        doc_chunk_ids = []
        if all_doc_ids:
            try:
                cursor = self.vdb.conn.cursor()
                placeholders = ",".join("?" for _ in all_doc_ids)
                cursor.execute(f"SELECT id FROM chunks WHERE document_id IN ({placeholders})", all_doc_ids)
                doc_chunk_ids = [row[0] for row in cursor.fetchall()]
            except Exception as e:
                print(f"Error fetching chunk IDs for documents: {e}")

        # 3. Fetch chunk IDs for direct concepts and their 1-hop neighbors in the graph
        concept_chunk_ids = self.gdb.get_neighborhood_chunk_ids(concept_ids)
        
        # Combine concept chunks and document chunks (prioritizing document chunks if document/project was queried)
        graph_chunk_ids = []
        seen_chunks = set()
        
        for cid in doc_chunk_ids:
            if cid not in seen_chunks:
                graph_chunk_ids.append(cid)
                seen_chunks.add(cid)
                
        for cid in concept_chunk_ids:
            if cid not in seen_chunks:
                graph_chunk_ids.append(cid)
                seen_chunks.add(cid)

        # Limit to top_k_graph results to keep context size manageable
        graph_chunk_ids = graph_chunk_ids[:top_k_graph]

        # 4. PASS 2: Retrieve chunks from ChromaDB by IDs
        graph_chunks = []
        if graph_chunk_ids:
            graph_chunks = self.vdb.get_chunks_by_ids(graph_chunk_ids)

        # 5. PASS 3: Semantic Vector Search
        semantic_chunks = self.vdb.query_vector(query, n_results=top_k_vector)
        # Filter semantic chunks by a similarity threshold (0.45 cosine similarity) to block completely unrelated files
        semantic_chunks = [c for c in semantic_chunks if c.get("similarity", 0.0) >= 0.45]

        # 6. MERGE & FUSION
        # We merge chunks by ID. If a chunk exists in both, we combine and prioritize.
        merged_chunks = []
        seen_ids = set()
        
        # Priority 1: Chunks that matched BOTH Graph concept keywords and Vector search
        semantic_ids = {c["id"] for c in semantic_chunks}
        graph_ids = {c["id"] for c in graph_chunks}
        
        both_ids = semantic_ids.intersection(graph_ids)
        for chunk in semantic_chunks:
            if chunk["id"] in both_ids and chunk["id"] not in seen_ids:
                merged_chunks.append(chunk)
                seen_ids.add(chunk["id"])
                
        # Priority 2: Chunks from Graph keyword mapping
        for chunk in graph_chunks:
            if chunk["id"] not in seen_ids:
                merged_chunks.append(chunk)
                seen_ids.add(chunk["id"])

        # Priority 3: Chunks from Vector semantic search
        for chunk in semantic_chunks:
            if chunk["id"] not in seen_ids:
                merged_chunks.append(chunk)
                seen_ids.add(chunk["id"])

        # 7. GET RELATIONSHIP CONTEXT
        # Query Graph DB for connections between the concept nodes we found
        relationships = self.gdb.get_connected_subgraph(concept_ids)
        
        # Format relations as text lines
        rel_lines = []
        for rel in relationships:
            rel_lines.append(f"- ({rel['source']}) -[{rel['type']}]-> ({rel['target']}) | İlişki Açıklaması: {rel['description']}")

        # Gather organizational metadata for unique documents referenced
        doc_ids = set()
        for chunk in merged_chunks:
            d_id = chunk["metadata"].get("document_id")
            if d_id:
                doc_ids.add(d_id)
                
        doc_metadata_blocks = []
        for d_id in doc_ids:
            doc_meta = self.gdb.get_document_metadata(d_id)
            if doc_meta:
                meta_block = (
                    f"- Doküman Adı: {doc_meta['doc_name']}\n"
                    f"  - Türü: {doc_meta['doc_type']}\n"
                    f"  - Yükleyen Kişi: {doc_meta['user_name']}\n"
                    f"  - Yükleme Zamanı: {doc_meta['upload_time']}\n"
                    f"  - Proje: {doc_meta['project_name']}\n"
                    f"  - Şirket: {doc_meta['company_name']}"
                )
                doc_metadata_blocks.append(meta_block)

        # Compile final context strings
        context_chunks_str = ""
        sources = []
        for idx, chunk in enumerate(merged_chunks):
            doc_name = chunk["metadata"].get("document_name", "Bilinmeyen Doküman")
            chunk_idx = chunk["metadata"].get("chunk_index", 0)
            context_chunks_str += f"[KAYNAK {idx+1}: {doc_name} (Bölüm {chunk_idx})]\n{chunk['text']}\n\n"
            sources.append(f"{doc_name} (Bölüm {chunk_idx})")
            
        sources = list(set(sources))

        context_relations_str = ""
        if rel_lines:
            context_relations_str = "Veritabanından Bulunan Kavram İlişkileri:\n" + "\n".join(rel_lines)

        full_context = ""
        if doc_metadata_blocks:
            full_context += "DOKÜMAN ORGANİZASYON METAVERİLERİ (KİM, NE ZAMAN, HANGİ PROJE):\n" + "\n".join(doc_metadata_blocks) + "\n\n"
        if context_chunks_str:
            full_context += "DOKÜMAN PARÇALARI:\n" + context_chunks_str
        if context_relations_str:
            full_context += "\n" + context_relations_str
        if concepts_metadata:
            full_context += "\nİLGİLİ KAVRAM TANIMLARI:\n" + "\n".join(concepts_metadata)

        # Hallucination Guard: If no context was retrieved, return empty routing to prevent LLM hallucinating
        if not full_context.strip():
            return {
                "routing": "empty",
                "context_text": "",
                "chunks": [],
                "relationships": [],
                "concepts_found": [],
                "sources": []
            }

        return {
            "routing": "hybrid",
            "context_text": full_context.strip(),
            "chunks": merged_chunks,
            "relationships": relationships,
            "concepts_found": graph_concepts,
            "sources": sources
        }
