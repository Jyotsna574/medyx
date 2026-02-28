# Neo4j Setup for MedicalAgentDiagnosis

## 1. Connection (Neo4j on Your Desktop)

When the pipeline runs on the **same machine** as Neo4j Desktop:

```bash
# Create .env from example
cp .env.example .env

# Edit .env - set your Neo4j password (from Neo4j Desktop when you created the DB)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your_password>
```

**Verify**: In Neo4j Browser (`http://localhost:7474`), run `:server connect` and confirm the database is running.

## 2. Required Graph Schema

The retriever expects:

- **Entity** nodes with:
  - `type`: `disease` | `drug` | `gene/protein` | `effect/phenotype`
  - `name`: string (searchable)
  - `source`: optional string
- **RELATED** relationships between disease ↔ drug and disease ↔ gene

If your database is empty or uses a different schema, queries will return nothing and raise `Neo4jQueryError`.

## 3. Quick Schema + Sample Data

Run this in Neo4j Browser to create the schema and some test data:

```cypher
// Create sample disease entities
CREATE (g:Entity {name: "Glaucoma", type: "disease", source: "ICD-10"})
CREATE (p:Entity {name: "Pneumonia", type: "disease", source: "ICD-10"})
CREATE (t:Entity {name: "Tuberculosis", type: "disease", source: "ICD-10"});

// Create drugs
CREATE (t1:Entity {name: "Timolol", type: "drug"})
CREATE (t2:Entity {name: "Latanoprost", type: "drug"})
CREATE (a:Entity {name: "Amoxicillin", type: "drug"});

// Relationships
MATCH (g:Entity {name: "Glaucoma"}), (t1:Entity {name: "Timolol"})
CREATE (g)-[:RELATED]->(t1);
MATCH (g:Entity {name: "Glaucoma"}), (t2:Entity {name: "Latanoprost"})
CREATE (g)-[:RELATED]->(t2);
MATCH (p:Entity {name: "Pneumonia"}), (a:Entity {name: "Amoxicillin"})
CREATE (p)-[:RELATED]->(a);
```

## 4. Run from Another Machine

If the pipeline runs on a server/cluster and Neo4j is on your desktop:

1. Neo4j must accept remote connections (listen on `0.0.0.0`).
2. Open port 7687 in your desktop firewall.
3. Set `NEO4J_URI=bolt://<your-desktop-ip>:7687`.
