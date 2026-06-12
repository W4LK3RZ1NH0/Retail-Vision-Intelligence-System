# Retail-Vision-Intelligence-System
Intelligent retail shelf monitoring system using Multimodal LLMs for real-time inspection, natural language rule processing, and historical RAG-based analysis. Developed for the LIACD 2025/2026 course.

## RAG Chunking Strategy

The system uses a **hybrid chunking strategy** (summary + metadata) as specified in §6.5.

### Why hybrid?
- **Single-record chunking** is simple but the embedding becomes an average of all content, losing granularity.
- **Per-issue chunking** enables granular retrieval but increases index size and fragment context.
- **Hybrid** combines the best of both: the inspection summary serves as the searchable text (semantic richness), while structured metadata (zone, date, fill_rate, status, issue_types) enables pre-retrieval filtering.

### How it works
Each inspection generates one chunk with:
- `content`: Rich summary generated from inspection data (zone, products, issues, fill rate, reasoning)
- `metadata`: Structured fields for filtering before retrieval:
  - `source`: `shelf_inspection`, `rule_violation`, or `tp1_trajectory`
  - `zone_id`: e.g., `Z_S1`
  - `timestamp`: ISO datetime
  - `fill_rate`: 0.0–1.0
  - `overall_status`: `ok`, `warning`, `critical`
  - `issue_types`: list of detected issue categories

### Trade-offs
| Strategy | Pros | Cons |
|----------|------|------|
| Single record | Simple, fast indexing | Poor granularity |
| Per-issue | Granular retrieval | Large index, fragmented context |
| Hybrid (chosen) | Semantic search + metadata filtering | Slightly more complex generation |

### Retrieval
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Vector store: ChromaDB (persistent, local)
- Top-k: 3 (configurable)
- Pre-filtering: metadata filters applied before similarity search when zone/date/source constraints are provided.
