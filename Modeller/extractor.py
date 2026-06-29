import os
import re
import json
import datetime
import fitz  # PyMuPDF for robust text extraction
import ollama
from Veritabanı.vector_db import VectorDBManager
from Veritabanı.graph_db import GraphDBManager

class DocumentExtractor:
    def __init__(self, vector_db: VectorDBManager, graph_db: GraphDBManager, model_name: str = "qwen3:4b"):
        self.vdb = vector_db
        self.gdb = graph_db
        self.model_name = model_name

    def read_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF (extremely fast & robust)."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_path}")
        
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text += page_text + "\n"
        doc.close()
        return text

    def chunk_text(self, text: str, chunk_size: int = 1000) -> list:
        """
        Rule-based chunking that respects paragraph boundaries (\n\n) 
        and matches ~chunk_size words.
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            words_in_para = len(para.split())
            
            if words_in_para > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    sentence_words = len(sentence.split())
                    if current_word_count + sentence_words > chunk_size and current_chunk:
                        chunks.append("\n".join(current_chunk))
                        current_chunk = []
                        current_word_count = 0
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
            else:
                if current_word_count + words_in_para > chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                current_chunk.append(para)
                current_word_count += words_in_para
                
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return chunks

    def extract_concepts_and_relations(self, chunk_text: str) -> dict:
        """
        Call Ollama model to extract key concepts (with types) and relationships in JSON format.
        """
        system_prompt = """
        IDENTITY:
        You are an AI pipeline component that extracts knowledge graphs from text.
        
        PURPOSE:
        Analyze the text chunk and extract key technical concepts, entities, and the relationships between them.
        
        STRICT RULES:
        1. OUTPUT FORMAT: You must return ONLY a valid JSON object matching the JSON schema below. Use no markdown code blocks.
        2. LANGUAGE: Understand the text (which could be in Turkish or English) and output all concept names, types, descriptions, and relationship descriptions in English.
        3. CONCEPT LIMIT: Extract at most 8 core concepts per chunk.
        4. RELATIONSHIP LIMIT: Extract at most 8 key relationships.
        5. CONCISENESS: Keep descriptions very short (maximum 8 words).
        6. CONCEPT TYPE: For each concept, assign a type from this list: 'Technology', 'Person', 'Algorithm', 'Parameter', 'Other'.
        
        JSON SCHEMA:
        {
          "concepts": [
            {
              "name": "Concept name (e.g. 'Microgrid Optimization', 'Ollama')",
              "type": "Concept type (must be one of: 'Technology', 'Person', 'Algorithm', 'Parameter', 'Other')",
              "description": "Short explanation of the concept"
            }
          ],
          "relations": [
            {
              "source": "Name of source concept",
              "target": "Name of target concept",
              "type": "Relationship type (e.g. 'IMPLEMENTS', 'USED_BY', 'DISCUSSES', 'OPTIMIZES')",
              "description": "Short explanation of how they are related"
            }
          ]
        }
        """

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"TEXT CHUNK:\n{chunk_text}"}
                ],
                format="json",
                options={"temperature": 0.1, "num_predict": 1024}
            )
            
            content = response["message"]["content"]
            data = json.loads(content)
            return data
        except Exception as e:
            print(f"Extraction error using {self.model_name}: {e}")
            return {"concepts": [], "relations": []}

    def ingest_document(self, pdf_path: str, doc_metadata: dict = None, progress_callback=None) -> dict:
        """
        Full ingestion pipeline:
        1. Extract text from PDF.
        2. Create User, Project, Company, Document nodes in Neo4j and link them.
        3. Chunk text.
        4. Index chunks in SQLite.
        5. Extract concepts & relations from each chunk using Ollama.
        6. Add nodes and relationships to Neo4j.
        """
        doc_name = os.path.basename(pdf_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"doc_{timestamp}"

        # 1. Read PDF
        if progress_callback:
            progress_callback("PDF okunuyor...", 5)
        text = self.read_pdf(pdf_path)
        
        if not text.strip():
            raise ValueError("PDF dosyasından okunabilir metin çıkarılamadı.")

        # Copy the original file to a local Documents directory within the workspace
        try:
            # The project root is one level up from 'Modeller'
            project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            docs_dir = os.path.join(project_dir, "Documents")
            os.makedirs(docs_dir, exist_ok=True)
            saved_file_path = os.path.join(docs_dir, doc_name)
            
            import shutil
            shutil.copy2(pdf_path, saved_file_path)
        except Exception as e:
            print(f"Error copying document to workspace folder: {e}")
            saved_file_path = pdf_path # Fallback to original path if copy fails

        # 2. Add Organization structure nodes in Neo4j
        if progress_callback:
            progress_callback("Organizasyon yapısı oluşturuluyor...", 10)
            
        if doc_metadata:
            user_name = doc_metadata.get("user", "System").strip() or "System"
            project_name = doc_metadata.get("project", "Default Project").strip() or "Default Project"
            doc_type = doc_metadata.get("doc_type", "Dokümantasyon").strip() or "Dokümantasyon"
            company_name = doc_metadata.get("company", "Default Company").strip() or "Default Company"
        else:
            user_name = "System"
            project_name = "Default Project"
            doc_type = "Dokümantasyon"
            company_name = "Default Company"

        self.gdb.add_org_nodes(
            doc_id=doc_id, 
            doc_name=doc_name, 
            doc_type=doc_type, 
            user_name=user_name, 
            project_name=project_name, 
            company_name=company_name,
            file_path=saved_file_path
        )

        # 3. Chunking
        if progress_callback:
            progress_callback("Metin parçalara bölünüyor...", 15)
        chunks = self.chunk_text(text)
        total_chunks = len(chunks)

        if total_chunks == 0:
            raise ValueError("Metin boş veya çok kısa, parçalanamadı.")

        # 4. For each chunk, index & extract
        all_chunk_ids = []
        concepts_to_add = []
        relationships_to_add = []
        
        for i, chunk in enumerate(chunks):
            progress_percent = 15 + int((i / total_chunks) * 80)
            if progress_callback:
                progress_callback(f"Parça {i+1}/{total_chunks} işleniyor ve analiz ediliyor...", progress_percent)

            # A. Add to SQLite
            chunk_id = self.vdb.add_chunk(doc_id=doc_id, chunk_index=i, text=chunk, doc_name=doc_name)
            all_chunk_ids.append(chunk_id)

            # B. Extract concepts and relations
            graph_data = self.extract_concepts_and_relations(chunk)
            
            # C. Accumulate extracted concepts and document links
            for concept in graph_data.get("concepts", []):
                concept_name = concept.get("name", "").strip()
                concept_type = concept.get("type", "Other").strip()
                concept_desc = concept.get("description", "").strip()
                
                # Normalize concept type to valid options
                if concept_type not in ["Technology", "Person", "Algorithm", "Parameter", "Other"]:
                    concept_type = "Other"

                if concept_name:
                    concepts_to_add.append({
                        "name": concept_name,
                        "description": concept_desc,
                        "concept_type": concept_type,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id
                    })
                    
                    # Accumulate document-to-concept relationship
                    relationships_to_add.append({
                        "source_name": doc_name,
                        "target_name": concept_name,
                        "rel_type": "DISCUSSES",
                        "description": f"Discussed in {doc_name}",
                        "source_label": "Document",
                        "target_label": "Concept"
                    })

            # D. Accumulate concept relationships
            for relation in graph_data.get("relations", []):
                source = relation.get("source", "").strip()
                target = relation.get("target", "").strip()
                rel_type = relation.get("type", "RELATES_TO").strip()
                rel_desc = relation.get("description", "").strip()
                
                if source and target:
                    # Ensure both nodes exist first by adding them to concept batch
                    concepts_to_add.append({
                        "name": source,
                        "description": source,
                        "concept_type": "Other",
                        "doc_id": doc_id,
                        "chunk_id": chunk_id
                    })
                    concepts_to_add.append({
                        "name": target,
                        "description": target,
                        "concept_type": "Other",
                        "doc_id": doc_id,
                        "chunk_id": chunk_id
                    })
                    # Accumulate relation
                    relationships_to_add.append({
                        "source_name": source,
                        "target_name": target,
                        "rel_type": rel_type,
                        "description": rel_desc,
                        "source_label": "Concept",
                        "target_label": "Concept"
                    })

        # E. Write all concepts and relationships in batch to Neo4j
        if progress_callback:
            progress_callback("İlişkiler ve kavramlar veritabanına toplu olarak kaydediliyor...", 95)
        self.gdb.add_concepts_batch(concepts_to_add)
        self.gdb.add_relationships_batch(relationships_to_add)

        # Update global SystemMetadata node with this latest upload's registry and chunks
        upload_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.gdb.update_system_metadata(
            doc_id=doc_id,
            doc_name=doc_name,
            user_name=user_name,
            project_name=project_name,
            company_name=company_name,
            upload_time=upload_time_str,
            chunk_ids=all_chunk_ids
        )

        if progress_callback:
            progress_callback("Dizin oluşturma tamamlandı!", 100)

        return {
            "document_id": doc_id,
            "document_name": doc_name,
            "total_chunks": total_chunks,
            "timestamp": timestamp,
            "metadata": {
                "user": user_name,
                "project": project_name,
                "doc_type": doc_type,
                "company": company_name
            }
        }
