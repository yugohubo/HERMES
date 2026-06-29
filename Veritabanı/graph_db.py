from neo4j import GraphDatabase
import re

class Neo4jConnectionError(Exception):
    """Custom exception raised when Neo4j database cannot be reached."""
    def __init__(self, uri, message="Neo4j veritabanı bağlantısı kurulamadı. Lütfen veritabanının açık olduğunu kontrol edin."):
        self.uri = uri
        self.message = f"{message} (Adres: {uri})"
        super().__init__(self.message)

class GraphDBManager:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.connect()

    def connect(self):
        """Establish connection to Neo4j. Raises Neo4jConnectionError on failure."""
        try:
            # Suppress empty database warnings/notifications in console
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                notifications_min_severity="OFF"
            )
            # Test the connection with a quick query
            self.driver.verify_connectivity()
        except Exception as e:
            if self.driver:
                self.driver.close()
            raise Neo4jConnectionError(self.uri, str(e))

    def close(self):
        if self.driver:
            self.driver.close()

    def normalize_id(self, text: str) -> str:
        """Helper to create a clean, lowercase ID for nodes."""
        text = text.lower().strip()
        text = re.sub(r'[\s\-_]+', '_', text)
        text = re.sub(r'[^a-z0-9_]', '', text)
        return text

    def add_concept_node(self, name: str, description: str, doc_id: str, chunk_id: str, concept_type: str = "Other"):
        """Add a concept node to the graph and merge if it already exists."""
        node_id = self.normalize_id(name)
        
        query = """
        MERGE (c:Concept {id: $id})
        ON CREATE SET c.name = $name, 
                      c.description = $description,
                      c.type = $concept_type,
                      c.doc_id = $doc_id,
                      c.chunk_ids = [$chunk_id]
        ON MATCH SET c.chunk_ids = CASE WHEN NOT $chunk_id IN c.chunk_ids THEN c.chunk_ids + $chunk_id ELSE c.chunk_ids END,
                     c.description = CASE WHEN c.description = '' OR c.description IS NULL OR c.description = $name THEN $description ELSE c.description END,
                     c.type = CASE WHEN c.type = 'Other' OR c.type IS NULL THEN $concept_type ELSE c.type END
        RETURN c
        """
        
        with self.driver.session() as session:
            session.run(query, id=node_id, name=name, description=description, doc_id=doc_id, chunk_id=chunk_id, concept_type=concept_type)
        
        # Automatically resolve cross-document synonyms/related names
        self.link_similar_concepts(name)

    def link_similar_concepts(self, new_concept_name: str):
        """Automatically create RELATES_TO links to other concepts with similar names in the graph."""
        new_id = self.normalize_id(new_concept_name)
        if len(new_id) < 3:
            return
            
        query = """
        MATCH (c:Concept)
        WHERE c.id <> $new_id AND (
            c.id CONTAINS $new_id 
            OR $new_id CONTAINS c.id
        )
        MATCH (new:Concept {id: $new_id})
        MERGE (new)-[r:RELATES_TO {description: "İsim benzerliği nedeniyle otomatik bağlandı"}]-(c)
        """
        try:
            with self.driver.session() as session:
                session.run(query, new_id=new_id)
        except Exception:
            pass

    def add_concepts_batch(self, concepts: list):
        """Add concept nodes to the graph in bulk using a single Neo4j transaction."""
        if not concepts:
            return
            
        query = """
        UNWIND $concepts AS c_data
        MERGE (c:Concept {id: c_data.id})
        ON CREATE SET c.name = c_data.name, 
                      c.description = c_data.description,
                      c.type = c_data.concept_type,
                      c.doc_id = c_data.doc_id,
                      c.chunk_ids = [c_data.chunk_id]
        ON MATCH SET c.chunk_ids = CASE WHEN NOT c_data.chunk_id IN c.chunk_ids THEN c.chunk_ids + c_data.chunk_id ELSE c.chunk_ids END,
                     c.description = CASE WHEN c.description = '' OR c.description IS NULL OR c.description = c_data.name THEN c_data.description ELSE c.description END,
                     c.type = CASE WHEN c.type = 'Other' OR c.type IS NULL THEN c_data.concept_type ELSE c.type END
        """
        batch_params = []
        for c in concepts:
            name = c.get("name", "").strip()
            if not name:
                continue
            batch_params.append({
                "id": self.normalize_id(name),
                "name": name,
                "description": c.get("description", "").strip(),
                "concept_type": c.get("concept_type", "Other").strip(),
                "doc_id": c.get("doc_id", ""),
                "chunk_id": c.get("chunk_id", "")
            })
            
        if not batch_params:
            return
            
        try:
            with self.driver.session() as session:
                session.run(query, concepts=batch_params)
            
            # Link similar concepts for each concept name in batch
            for c in concepts:
                name = c.get("name", "").strip()
                if name:
                    self.link_similar_concepts(name)
        except Exception as e:
            print(f"Error in add_concepts_batch: {e}")

    def add_relationships_batch(self, rels: list):
        """Add relationships in batch, grouped by labels and type to optimize transactions."""
        if not rels:
            return
            
        # Group relationships by (source_label, target_label, rel_type_clean)
        groups = {}
        for r in rels:
            src_lbl = r.get("source_label", "Concept")
            tgt_lbl = r.get("target_label", "Concept")
            rel_type = r.get("rel_type", "RELATES_TO").strip().upper()
            rel_type_clean = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type)
            if not rel_type_clean:
                rel_type_clean = "RELATES_TO"
                
            key = (src_lbl, tgt_lbl, rel_type_clean)
            if key not in groups:
                groups[key] = []
                
            groups[key].append({
                "source_id": self.normalize_id(r["source_name"]),
                "target_id": self.normalize_id(r["target_name"]),
                "description": r.get("description", "")
            })
            
        # Execute batch for each group
        with self.driver.session() as session:
            for (src_lbl, tgt_lbl, rel_type), batch in groups.items():
                query = f"""
                UNWIND $batch AS r_data
                MATCH (source:{src_lbl} {{id: r_data.source_id}})
                MATCH (target:{tgt_lbl} {{id: r_data.target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                ON CREATE SET r.description = r_data.description
                ON MATCH SET r.description = CASE WHEN r.description = '' OR r.description IS NULL THEN r_data.description ELSE r.description END
                """
                try:
                    session.run(query, batch=batch)
                except Exception as e:
                    print(f"Error in add_relationships_batch for {src_lbl}->{tgt_lbl} ({rel_type}): {e}")

    def get_neighborhood_chunk_ids(self, concept_ids: list) -> list:
        """Retrieve chunk IDs of the matching concepts and their 1-hop neighbor concepts in the graph."""
        if not concept_ids:
            return []
            
        query = """
        MATCH (c)
        WHERE (c:Concept OR c:SystemMetadata) AND c.id IN $concept_ids
        OPTIONAL MATCH (c)-[r]-(neighbor)
        WHERE neighbor:Concept OR neighbor:SystemMetadata
        RETURN c.chunk_ids AS direct_chunks, neighbor.chunk_ids AS neighbor_chunks
        """
        
        chunk_ids = []
        try:
            with self.driver.session() as session:
                res = session.run(query, concept_ids=concept_ids)
                for rec in res:
                    if rec["direct_chunks"]:
                        chunk_ids.extend(rec["direct_chunks"])
                    if rec["neighbor_chunks"]:
                        chunk_ids.extend(rec["neighbor_chunks"])
        except Exception as e:
            print(f"Error fetching neighborhood chunk IDs: {e}")
            
        cleaned_ids = []
        for cid in chunk_ids:
            if cid and cid != "doc_node" and cid not in cleaned_ids:
                cleaned_ids.append(cid)
        return cleaned_ids

    def add_org_nodes(self, doc_id: str, doc_name: str, doc_type: str, user_name: str, project_name: str, company_name: str, file_path: str = ""):
        """
        Create and link organizational structure nodes:
        (User) -[:UPLOADED]-> (Document) -[:BELONGS_TO]-> (Project) -[:OWNED_BY]-> (Company)
        """
        u_id = self.normalize_id(user_name)
        p_id = self.normalize_id(project_name)
        c_id = self.normalize_id(company_name)

        query = """
        // 1. Merge Company
        MERGE (co:Company {id: $c_id})
        ON CREATE SET co.name = $company_name

        // 2. Merge Project and Link to Company
        MERGE (pr:Project {id: $p_id})
        ON CREATE SET pr.name = $project_name
        MERGE (pr)-[:OWNED_BY]->(co)

        // 3. Merge Document and Link to Project
        MERGE (doc:Document {id: $doc_id})
        ON CREATE SET doc.name = $doc_name, doc.type = $doc_type, doc.file_path = $file_path
        ON MATCH SET doc.file_path = $file_path
        MERGE (doc)-[:BELONGS_TO]->(pr)

        // 4. Merge User and Link to Document
        MERGE (u:User {id: $u_id})
        ON CREATE SET u.name = $user_name
        MERGE (u)-[:UPLOADED {timestamp: datetime()}]->(doc)
        """
        
        with self.driver.session() as session:
            session.run(query, 
                        doc_id=doc_id, doc_name=doc_name, doc_type=doc_type,
                        user_name=user_name, u_id=u_id,
                        project_name=project_name, p_id=p_id,
                        company_name=company_name, c_id=c_id,
                        file_path=file_path)

    def update_system_metadata(self, doc_id: str, doc_name: str, user_name: str, project_name: str, company_name: str, upload_time: str, chunk_ids: list):
        """Update the global SystemMetadata node with the latest upload details."""
        query = """
        MERGE (m:SystemMetadata {id: "global_metadata"})
        SET m.name = "En Son Yüklenen Veri (Latest Uploaded Data)",
            m.description = "Sistemdeki en son yüklenen dosya: " + $doc_name + ". Yükleyen: " + $user_name + ", Zaman: " + $upload_time + ", Proje: " + $project_name + ", Şirket: " + $company_name,
            m.latest_doc_id = $doc_id,
            m.latest_doc_name = $doc_name,
            m.latest_uploader = $user_name,
            m.latest_upload_time = $upload_time,
            m.latest_project = $project_name,
            m.latest_company = $company_name,
            m.chunk_ids = $chunk_ids
        """
        try:
            with self.driver.session() as session:
                session.run(query, doc_id=doc_id, doc_name=doc_name, user_name=user_name, 
                            project_name=project_name, company_name=company_name, 
                            upload_time=upload_time, chunk_ids=chunk_ids)
        except Exception as e:
            print(f"Error updating system metadata: {e}")

    def add_relationship(self, source_name: str, target_name: str, rel_type: str, description: str, source_label: str = "Concept", target_label: str = "Concept"):
        """Add a relationship between two nodes with specified labels."""
        source_id = self.normalize_id(source_name)
        target_id = self.normalize_id(target_name)
        
        # Clean relation type to be Cypher-safe
        rel_type_clean = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type.upper().strip())
        if not rel_type_clean:
            rel_type_clean = "RELATES_TO"
 
        # Safe labels
        lbl_src = "Concept" if source_label not in ["Concept", "Document", "User", "Project", "Company"] else source_label
        lbl_tgt = "Concept" if target_label not in ["Concept", "Document", "User", "Project", "Company"] else target_label
 
        query = f"""
        MATCH (source:{lbl_src} {{id: $source_id}})
        MATCH (target:{lbl_tgt} {{id: $target_id}})
        MERGE (source)-[r:{rel_type_clean}]->(target)
        ON CREATE SET r.description = $description
        ON MATCH SET r.description = CASE WHEN r.description = '' OR r.description IS NULL THEN $description ELSE r.description END
        RETURN r
        """
        
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id, description=description)

    def search_nodes_by_keywords(self, keywords: list) -> list:
        """
        Search concepts, documents, or metadata matching keywords and return their associated chunk IDs.
        """
        if not keywords:
            return []
            
        query = """
        UNWIND $keywords AS keyword
        MATCH (c)
        WHERE (c:Concept OR c:Document OR c:Project OR c:User OR c:SystemMetadata) AND (
           toLower(c.name) CONTAINS toLower(keyword) 
           OR ((c:Concept OR c:SystemMetadata) AND toLower(c.description) CONTAINS toLower(keyword))
        )
        RETURN DISTINCT labels(c)[0] AS label, c.id AS id, c.name AS name, 
                        CASE WHEN c:Concept OR c:SystemMetadata THEN c.description ELSE '' END AS description,
                        CASE WHEN c:Concept OR c:SystemMetadata THEN c.chunk_ids ELSE [] END AS chunk_ids
        """
        
        results = []
        with self.driver.session() as session:
            res = session.run(query, keywords=keywords)
            for record in res:
                results.append({
                    "label": record["label"],
                    "id": record["id"],
                    "name": record["name"],
                    "description": record["description"],
                    "chunk_ids": record["chunk_ids"] or []
                })
        return results

    def get_connected_subgraph(self, node_ids: list):
        """Get relationships between specified nodes for context."""
        if not node_ids:
            return []
            
        query = """
        MATCH (s)-[r]->(t)
        WHERE s.id IN $ids AND t.id IN $ids
        RETURN labels(s)[0] AS s_label, s.name AS source, 
               labels(t)[0] AS t_label, t.name AS target, 
               type(r) AS rel_type, r.description AS description
        """
        
        relationships = []
        with self.driver.session() as session:
            res = session.run(query, ids=node_ids)
            for record in res:
                relationships.append({
                    "s_label": record["s_label"],
                    "source": record["source"],
                    "t_label": record["t_label"],
                    "target": record["target"],
                    "type": record["rel_type"],
                    "description": record["description"] or ""
                })
        return relationships

    def get_graph_data(self) -> dict:
        """Returns all nodes and edges in format suitable for frontend visualization."""
        nodes_query = """
        MATCH (n)
        WHERE n:Concept OR n:Document OR n:User OR n:Project OR n:Company
        RETURN labels(n)[0] AS label, n.id AS id, n.name AS name, 
               CASE WHEN n:Concept THEN n.description ELSE '' END AS description,
               CASE WHEN n:Concept THEN n.type ELSE 'Other' END AS concept_type
        """
        
        edges_query = """
        MATCH (s)-[r]->(t)
        WHERE (s:Concept OR s:Document OR s:User OR s:Project OR s:Company)
          AND (t:Concept OR t:Document OR t:User OR t:Project OR t:Company)
        RETURN s.id AS source, t.id AS target, type(r) AS type, r.description AS description
        """
        
        nodes = []
        edges = []
        
        with self.driver.session() as session:
            # Fetch nodes
            n_res = session.run(nodes_query)
            for rec in n_res:
                nodes.append({
                    "id": rec["id"],
                    "label": rec["name"],
                    "node_label": rec["label"], # User, Project, Company, Document, Concept
                    "description": rec["description"] or "",
                    "concept_type": rec["concept_type"]
                })
            # Fetch edges
            e_res = session.run(edges_query)
            for rec in e_res:
                edges.append({
                    "source": rec["source"],
                    "target": rec["target"],
                    "label": rec["type"],
                    "description": rec["description"] or ""
                })
                
        return {"nodes": nodes, "edges": edges}

    def get_system_summary(self) -> dict:
        """Compiles stats and node descriptions for Router context."""
        stats = self.get_stats()
        
        query = """
        MATCH (c:Concept)
        RETURN c.name AS name, c.description AS description, c.type AS type
        LIMIT 20
        """
        
        concepts = []
        with self.driver.session() as session:
            res = session.run(query)
            for rec in res:
                concepts.append(f"- {rec['name']} ({rec['type']}): {rec['description']}")
                
        concepts_str = "\n".join(concepts)
        
        summary_text = (
            f"Sistem Özet Bilgisi:\n"
            f"Toplam Kullanıcı: {stats['total_users']}\n"
            f"Toplam Şirket: {stats['total_companies']}\n"
            f"Toplam Proje: {stats['total_projects']}\n"
            f"Toplam Doküman: {stats['total_documents']}\n"
            f"Toplam Kavram: {stats['total_concepts']}\n"
            f"Toplam İlişki (Kenar): {stats['total_edges']}\n"
            f"Veritabanındaki Temel Kavramlar:\n{concepts_str}"
        )
        return {
            "summary": summary_text,
            "stats": stats
        }

    def clear_graph(self):
        """Clears all nodes and relationships in Neo4j."""
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session() as session:
            session.run(query)

    def get_stats(self) -> dict:
        """Returns counts for all node labels and relationships."""
        stats = {
            "total_users": 0,
            "total_companies": 0,
            "total_projects": 0,
            "total_documents": 0,
            "total_concepts": 0,
            "total_edges": 0
        }
        
        queries = {
            "total_users": "MATCH (n:User) RETURN count(n) AS count",
            "total_companies": "MATCH (n:Company) RETURN count(n) AS count",
            "total_projects": "MATCH (n:Project) RETURN count(n) AS count",
            "total_documents": "MATCH (n:Document) RETURN count(n) AS count",
            "total_concepts": "MATCH (n:Concept) RETURN count(n) AS count",
            "total_edges": "MATCH ()-[r]->() RETURN count(r) AS count"
        }
        
        try:
            with self.driver.session() as session:
                for key, query in queries.items():
                    res = session.run(query).single()
                    if res:
                        stats[key] = res["count"]
        except Exception:
            pass
        return stats

    def get_document_metadata(self, doc_id: str) -> dict:
        """Fetch organizational metadata for a document ID from Neo4j."""
        query = """
        MATCH (d:Document {id: $doc_id})
        OPTIONAL MATCH (u:User)-[rel:UPLOADED]->(d)
        OPTIONAL MATCH (d)-[:BELONGS_TO]->(p:Project)
        OPTIONAL MATCH (p)-[:OWNED_BY]->(c:Company)
        RETURN d.name AS doc_name, d.type AS doc_type, 
               u.name AS user_name, rel.timestamp AS upload_time,
               p.name AS project_name, c.name AS company_name
        """
        
        try:
            with self.driver.session() as session:
                res = session.run(query, doc_id=doc_id).single()
                if res:
                    time_val = res["upload_time"]
                    time_str = "Bilinmiyor"
                    if time_val:
                        try:
                            # Convert datetime string or object
                            time_str = str(time_val)
                            match = re.match(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})', time_str)
                            if match:
                                time_str = f"{match.group(1)} {match.group(2)}"
                        except Exception:
                            pass
                    
                    return {
                        "doc_name": res["doc_name"] or "Bilinmeyen",
                        "doc_type": res["doc_type"] or "Dokümantasyon",
                        "user_name": res["user_name"] or "System",
                        "upload_time": time_str,
                        "project_name": res["project_name"] or "Default Project",
                        "company_name": res["company_name"] or "Default Company"
                    }
        except Exception:
            pass
        return {}
