"""Neo4j Knowledge Graph Retriever for medical data.

Neo4j driver is imported lazily - if neo4j package is not installed,
connect() fails gracefully and fallback guidelines are used.
This allows running on clusters (e.g. Param Shakti) without Neo4j.
"""

import os
from typing import Optional


class Neo4jKnowledgeRetriever:
    """Retrieves medical knowledge from Neo4j graph database.
    
    This retriever queries a medical knowledge graph containing diseases,
    drugs, genes, symptoms, and their relationships.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize the Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (defaults to NEO4J_URI env var)
            username: Neo4j username (defaults to NEO4J_USERNAME env var)
            password: Neo4j password (defaults to NEO4J_PASSWORD env var)
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        
        self._driver = None
        self._connected = False
        
    def connect(self) -> bool:
        """Establish connection to Neo4j.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.username, self.password)
            )
            self._driver.verify_connectivity()
            self._connected = True
            return True
        except ImportError:
            print("Neo4j package not installed. Using fallback guidelines. Install with: pip install neo4j")
            self._connected = False
            return False
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            self._connected = False
            return False
    
    def close(self):
        """Close the Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self._connected
    
    async def search(self, query: str) -> str:
        """Search the knowledge graph for relevant medical information.
        
        Args:
            query: Natural language query about medical conditions.
            
        Returns:
            Formatted string with relevant medical knowledge.
        """
        if not self._connected:
            if not self.connect():
                return self._get_fallback_guidelines()
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Query the knowledge graph
        results = []
        
        with self._driver.session() as session:
            # 1. Find disease entities matching keywords
            disease_info = self._query_diseases(session, keywords)
            if disease_info:
                results.append(disease_info)
            
            # 2. Find related drugs/treatments
            drug_info = self._query_drugs(session, keywords)
            if drug_info:
                results.append(drug_info)
            
            # 3. Find related genes/proteins
            gene_info = self._query_genes(session, keywords)
            if gene_info:
                results.append(gene_info)
            
            # 4. Find related phenotypes/effects
            phenotype_info = self._query_phenotypes(session, keywords)
            if phenotype_info:
                results.append(phenotype_info)
        
        if results:
            return "\n\n".join(results)
        else:
            return self._get_fallback_guidelines()
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract medical keywords from the query."""
        # Common medical terms to search for
        medical_terms = [
            "glaucoma", "eye", "optic", "retina", "vision", "intraocular",
            "pressure", "iop", "cdr", "cup", "disc", "nerve", "field",
            "treatment", "drug", "symptom", "disease", "diagnosis"
        ]
        
        query_lower = query.lower()
        found_keywords = []
        
        for term in medical_terms:
            if term in query_lower:
                found_keywords.append(term)
        
        # Always include glaucoma for this application
        if "glaucoma" not in found_keywords:
            found_keywords.append("glaucoma")
        
        return found_keywords
    
    def _query_diseases(self, session, keywords: list[str]) -> str:
        """Query disease entities from the graph."""
        keyword_patterns = "|".join(keywords)
        
        result = session.run("""
            MATCH (d:Entity)
            WHERE d.type = 'disease' 
              AND any(kw IN $keywords WHERE toLower(d.name) CONTAINS kw)
            RETURN d.name as name, d.source as source
            LIMIT 10
        """, keywords=keywords)
        
        diseases = list(result)
        if not diseases:
            return ""
        
        output = ["DISEASE ENTITIES FROM KNOWLEDGE GRAPH:", "-" * 40]
        for d in diseases:
            output.append(f"• {d['name']} (Source: {d['source']})")
        
        return "\n".join(output)
    
    def _query_drugs(self, session, keywords: list[str]) -> str:
        """Query drug relationships from the graph."""
        result = session.run("""
            MATCH (disease:Entity)-[r:RELATED]-(drug:Entity)
            WHERE disease.type = 'disease' 
              AND drug.type = 'drug'
              AND any(kw IN $keywords WHERE toLower(disease.name) CONTAINS kw)
            RETURN DISTINCT disease.name as disease, drug.name as drug
            LIMIT 25
        """, keywords=keywords)
        
        drugs = list(result)
        if not drugs:
            return ""
        
        output = ["RELATED DRUGS/TREATMENTS:", "-" * 40]
        
        # Group by disease
        drug_by_disease = {}
        for d in drugs:
            disease = d['disease']
            if disease not in drug_by_disease:
                drug_by_disease[disease] = []
            drug_by_disease[disease].append(d['drug'])
        
        for disease, drug_list in drug_by_disease.items():
            output.append(f"\n{disease}:")
            for drug in drug_list[:10]:  # Limit per disease
                output.append(f"  • {drug}")
        
        return "\n".join(output)
    
    def _query_genes(self, session, keywords: list[str]) -> str:
        """Query gene/protein relationships from the graph."""
        result = session.run("""
            MATCH (disease:Entity)-[r:RELATED]-(gene:Entity)
            WHERE disease.type = 'disease' 
              AND gene.type = 'gene/protein'
              AND any(kw IN $keywords WHERE toLower(disease.name) CONTAINS kw)
            RETURN DISTINCT disease.name as disease, gene.name as gene
            LIMIT 20
        """, keywords=keywords)
        
        genes = list(result)
        if not genes:
            return ""
        
        output = ["ASSOCIATED GENES/PROTEINS:", "-" * 40]
        
        # Group by disease
        gene_by_disease = {}
        for g in genes:
            disease = g['disease']
            if disease not in gene_by_disease:
                gene_by_disease[disease] = []
            gene_by_disease[disease].append(g['gene'])
        
        for disease, gene_list in gene_by_disease.items():
            output.append(f"\n{disease}:")
            output.append(f"  Genes: {', '.join(gene_list[:10])}")
        
        return "\n".join(output)
    
    def _query_phenotypes(self, session, keywords: list[str]) -> str:
        """Query phenotype/effect relationships from the graph."""
        result = session.run("""
            MATCH (d:Entity)
            WHERE d.type = 'effect/phenotype'
              AND any(kw IN $keywords WHERE toLower(d.name) CONTAINS kw)
            RETURN d.name as phenotype, d.source as source
            LIMIT 10
        """, keywords=keywords)
        
        phenotypes = list(result)
        if not phenotypes:
            return ""
        
        output = ["CLINICAL PHENOTYPES/EFFECTS:", "-" * 40]
        for p in phenotypes:
            output.append(f"• {p['phenotype']} (Source: {p['source']})")
        
        return "\n".join(output)
    
    def _get_fallback_guidelines(self) -> str:
        """Return fallback clinical guidelines when KG is not available."""
        return """
AAO (American Academy of Ophthalmology) Glaucoma Guidelines:

1. CUP-TO-DISC RATIO (CDR) ASSESSMENT:
   - CDR < 0.5: Generally considered normal
   - CDR 0.5-0.6: Borderline, requires monitoring
   - CDR > 0.6: Suspicious for glaucoma, further evaluation recommended
   - CDR > 0.7: High suspicion for glaucomatous damage

2. RISK FACTORS TO CONSIDER:
   - Family history of glaucoma
   - Age > 40 years
   - African or Hispanic ancestry
   - Elevated intraocular pressure (IOP)
   - Thin central corneal thickness
   - Myopia (nearsightedness)

3. DIAGNOSTIC CRITERIA:
   - Characteristic optic nerve damage (increased CDR, rim thinning)
   - Corresponding visual field defects
   - Open anterior chamber angle (for POAG)
   - Exclusion of secondary causes

4. MANAGEMENT RECOMMENDATIONS:
   - Mild (CDR 0.5-0.6): Monitor every 6-12 months
   - Moderate (CDR 0.6-0.7): Consider treatment, monitor every 3-6 months
   - Severe (CDR > 0.7): Immediate treatment initiation recommended
"""
