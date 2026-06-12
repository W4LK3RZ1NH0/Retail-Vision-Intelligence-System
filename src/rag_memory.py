#!/usr/bin/env python3
"""
RAG Memory - Semantic memory for retail shelf inspections using ChromaDB and sentence-transformers.
Implements hybrid chunking (summary + metadata) with top-k=3 retrieval.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()


class SourceType(str, Enum):
    SHELF_INSPECTION = "shelf_inspection"
    RULE_VIOLATION = "rule_violation"
    TP1_TRAJECTORY = "tp1_trajectory"
    REPORT = "report"


class RetrievalResult(BaseModel):
    """Result from vector similarity search."""
    id: str
    content: str
    metadata: dict[str, Any]
    distance: float
    score: float


class RAGMemory:
    def __init__(
        self,
        persist_dir: str = "./vectorstore",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_name: str = "gemini-1.5-flash",
        top_k: int = 3
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.model_name = model_name
        
        # Inicializa o cliente Gemini unificado
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY não configurado no ficheiro .env.")
        self.genai_client = genai.Client(api_key=self.api_key)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create/get collections
        self.collections = {}
        for coll_name in ["shelf_inspections", "rule_violations", "tp1_trajectory", "reports"]:
            self.collections[coll_name] = self.client.get_or_create_collection(
                name=coll_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # Load prompts
        prompts_dir = Path(__file__).parent.parent / "prompts"
        self.summary_prompt = (prompts_dir / "rag_summary.txt").read_text(encoding="utf-8") if (prompts_dir / "rag_summary.txt").exists() else ""
        self.query_prompt = (prompts_dir / "rag_query.txt").read_text(encoding="utf-8") if (prompts_dir / "rag_query.txt").exists() else ""
    
    def _generate_summary(self, inspection_data: dict[str, Any]) -> str:
        """Generate semantic summary using template."""
        issues = inspection_data.get("issues", [])
        issue_types = list(set(i.get("type", "") for i in issues if i.get("type")))
        severities = list(set(i.get("severity", "") for i in issues if i.get("severity")))
        locations = list(set(i.get("location", "") for i in issues if i.get("location")))
        
        summary_parts = [
            f"Zona {inspection_data.get('zone_id', 'N/A')}",
            f"fill rate {inspection_data.get('shelf_fill_rate', 0.0):.0%}",
            f"status {inspection_data.get('overall_status', 'N/A')}"
        ]
        
        if issue_types:
            summary_parts.append(f"issues: {', '.join(issue_types)}")
        if severities:
            summary_parts.append(f"severidades: {', '.join(severities)}")
        if locations:
            summary_parts.append(f"localizações: {', '.join(locations)}")
        
        summary_parts.append(f"data: {inspection_data.get('timestamp', '')[:10]}")
        
        return ". ".join(summary_parts)
    
    def _create_metadata(self, inspection_data: dict[str, Any], source: SourceType) -> dict[str, Any]:
        """Create structured metadata, keeping it compatible with ChromaDB primitive restrictions."""
        issues = inspection_data.get("issues", [])
        issue_types = ",".join(list(set(i.get("type", "") for i in issues if i.get("type"))))
        severities = ",".join(list(set(i.get("severity", "") for i in issues if i.get("severity"))))
        products = ",".join(inspection_data.get("products_detected", []))
        
        return {
            "source": source.value,
            "inspection_id": inspection_data.get("inspection_id", ""),
            "timestamp": inspection_data.get("timestamp", ""),
            "zone_id": inspection_data.get("zone_id", ""),
            "overall_status": inspection_data.get("overall_status", ""),
            "fill_rate": float(inspection_data.get("shelf_fill_rate", 0.0)),
            "issue_types": issue_types,  # Convertido para string simples
            "severities": severities,    # Convertido para string simples
            "products_detected": products  # Convertido para string simples
        }
    
    def add_inspection(self, inspection_data: dict[str, Any]) -> str:
        """Add a shelf inspection to the vector store."""
        chunk_id = inspection_data.get("inspection_id") or str(uuid.uuid4())
        
        summary = self._generate_summary(inspection_data)
        metadata = self._create_metadata(inspection_data, SourceType.SHELF_INSPECTION)
        embedding = self.embedding_model.encode(summary).tolist()
        
        self.collections["shelf_inspections"].add(
            ids=[chunk_id],
            documents=[summary],
            metadatas=[metadata],
            embeddings=[embedding]
        )
        return chunk_id
    
    def add_inspection_per_issue(self, inspection_data: dict[str, Any]) -> list[str]:
        """Per-issue chunking: one chunk per detected issue for granular retrieval."""
        parent_id = inspection_data.get("inspection_id") or str(uuid.uuid4())
        issues = inspection_data.get("issues", [])
        zone_id = inspection_data.get("zone_id", "")
        timestamp = inspection_data.get("timestamp", "")
        fill_rate = float(inspection_data.get("shelf_fill_rate", 0.0))
        products = inspection_data.get("products_detected", [])
        
        chunk_ids = []
        
        if not issues:
            chunk_id = self.add_inspection(inspection_data)
            return [chunk_id]
        
        for idx, issue in enumerate(issues):
            issue_type = issue.get("type", "other")
            severity = issue.get("severity", "low")
            location = issue.get("location", "")
            description = issue.get("description", "")
            
            content = (
                f"Zona {zone_id}, prateleira {location}. "
                f"Problema: {issue_type} (severidade {severity}). "
                f"{description}. "
                f"Fill rate global: {fill_rate:.0%}. "
                f"Produtos: {', '.join(products)}. "
                f"Data: {timestamp[:10]}."
            )
            
            metadata = {
                "source": SourceType.SHELF_INSPECTION.value,
                "inspection_id": parent_id,
                "chunk_type": "per_issue",
                "issue_index": idx,
                "timestamp": timestamp,
                "zone_id": zone_id,
                "fill_rate": fill_rate,
                "overall_status": inspection_data.get("overall_status", ""),
                "issue_types": issue_type,
                "severities": severity,
                "location": location,
                "products_detected": ",".join(products)
            }
            
            embedding = self.embedding_model.encode(content).tolist()
            chunk_id = f"{parent_id}_issue_{idx}"
            
            self.collections["shelf_inspections"].add(
                ids=[chunk_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            chunk_ids.append(chunk_id)
        
        return chunk_ids
    
    def add_rule_violation(self, violation_data: dict[str, Any]) -> str:
        """Add a rule violation to the vector store."""
        chunk_id = violation_data.get("rule_id", "") + "_" + violation_data.get("inspection_id", "")[:8]
        if not chunk_id or chunk_id == "_":
            chunk_id = str(uuid.uuid4())
        
        summary = f"Regra {violation_data.get('rule_id', 'N/A')} disparada na zona {violation_data.get('zone_id', 'N/A')}: {violation_data.get('notification', '')}"
        
        matched_issues = ",".join([i.get("type", "") for i in violation_data.get("matched_issues", []) if i.get("type")])
        
        metadata = {
            "source": SourceType.RULE_VIOLATION.value,
            "rule_id": violation_data.get("rule_id", ""),
            "inspection_id": violation_data.get("inspection_id", ""),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "zone_id": violation_data.get("zone_id", ""),
            "alert_level": violation_data.get("alert_level", ""),
            "matched_issues": matched_issues
        }
        
        embedding = self.embedding_model.encode(summary).tolist()
        self.collections["rule_violations"].add(
            ids=[chunk_id],
            documents=[summary],
            metadatas=[metadata],
            embeddings=[embedding]
        )
        return chunk_id
    
    def add_report(self, report_data: dict[str, Any]) -> str:
        """Add a generated report to the vector store."""
        chunk_id = report_data.get("report_id", str(uuid.uuid4()))
        summary = report_data.get("executive_summary", "Relatório de inspeção")
        
        zones = ",".join(report_data.get("zones_inspected", []))
        
        metadata = {
            "source": SourceType.REPORT.value,
            "report_id": chunk_id,
            "timestamp": report_data.get("timestamp", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
            "zones_inspected": zones,
            "critical_count": int(report_data.get("critical_count", 0)),
            "warning_count": int(report_data.get("warning_count", 0))
        }
        
        embedding = self.embedding_model.encode(summary).tolist()
        self.collections["reports"].add(
            ids=[chunk_id],
            documents=[summary],
            metadatas=[metadata],
            embeddings=[embedding]
        )
        return chunk_id

    def add_tp1_trajectory_data(self, tp1_data: dict[str, Any]) -> list[str]:
        """Add TP1 trajectory data (metrics, anomalies, zone performance) to vector store."""
        chunk_ids = []
        trafego = tp1_data.get("trafego", {})
        
        # Add traffic patterns per zone per day
        for day, visitors in trafego.get("unique_visitors_per_day", {}).items():
            chunk_id = f"tp1_traffic_{day}"
            summary = f"Tráfego diário {day}: {visitors} visitantes únicos na loja"
            metadata = {
                "source": SourceType.TP1_TRAJECTORY.value,
                "data_type": "daily_traffic",
                "date": day,
                "visitors": int(visitors)
            }
            embedding = self.embedding_model.encode(summary).tolist()
            self.collections["tp1_trajectory"].add(
                ids=[chunk_id], documents=[summary], metadatas=[metadata], embeddings=[embedding]
            )
            chunk_ids.append(chunk_id)
        
        # Add hourly traffic
        for hour, visitors in trafego.get("unique_visitors_per_hour", {}).items():
            chunk_id = f"tp1_traffic_hour_{hour}"
            summary = f"Tráfego horário {hour}h: {visitors} visitantes"
            metadata = {
                "source": SourceType.TP1_TRAJECTORY.value,
                "data_type": "hourly_traffic",
                "hour": int(hour),
                "visitors": int(visitors)
            }
            embedding = self.embedding_model.encode(summary).tolist()
            self.collections["tp1_trajectory"].add(
                ids=[chunk_id], documents=[summary], metadatas=[metadata], embeddings=[embedding]
            )
            chunk_ids.append(chunk_id)
        
        # Add zone traffic
        zona = tp1_data.get("zona", {})
        for zone_id, traffic in zona.get("traffic_per_zone", {}).items():
            chunk_id = f"tp1_zone_{zone_id}"
            summary = f"Zona {zone_id}: {traffic} visitantes totais"
            metadata = {
                "source": SourceType.TP1_TRAJECTORY.value,
                "data_type": "zone_traffic",
                "zone_id": zone_id,
                "traffic": int(traffic)
            }
            embedding = self.embedding_model.encode(summary).tolist()
            self.collections["tp1_trajectory"].add(
                ids=[chunk_id], documents=[summary], metadatas=[metadata], embeddings=[embedding]
            )
            chunk_ids.append(chunk_id)
        
        # Add anomalies
        anomalia = tp1_data.get("anomalia", {})
        for anomaly in anomalia.get("anomalies_detected", []):
            chunk_id = f"tp1_anomaly_{anomaly['zone_id']}_{anomaly['hour']}"
            summary = f"Anomalia {anomaly['direction']} na zona {anomaly['zone_id']} às {anomaly['hour']}h: valor {anomaly['day_7_value']} vs baseline {anomaly['baseline_mean']:.1f} (sigma={anomaly['sigma_distance']:.1f})"
            metadata = {
                "source": SourceType.TP1_TRAJECTORY.value,
                "data_type": "anomaly",
                "zone_id": anomaly["zone_id"],
                "hour": int(anomaly["hour"]),
                "direction": anomaly["direction"],
                "sigma_distance": float(anomaly["sigma_distance"]),
                "day_7_value": int(anomaly["day_7_value"]),
                "baseline_mean": float(anomaly["baseline_mean"]),
                "target_day": anomalia.get("target_day_evaluated", "")
            }
            embedding = self.embedding_model.encode(summary).tolist()
            self.collections["tp1_trajectory"].add(
                ids=[chunk_id], documents=[summary], metadatas=[metadata], embeddings=[embedding]
            )
            chunk_ids.append(chunk_id)
        
        return chunk_ids

    def query(
        self,
        natural_language_query: str,
        source_filter: list[SourceType] | None = None,
        zone_filter: list[str] | None = None,
        top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Query the vector store with natural language + metadata filters."""
        k = top_k or self.top_k
        
        # Configuração do filtro metadata com sintaxe segura ChromaDB
        where_conditions = []
        if source_filter:
            where_conditions.append({"source": {"$in": [s.value for s in source_filter]}})
        if zone_filter:
            where_conditions.append({"zone_id": {"$in": zone_filter}})
            
        where_clause = None
        if len(where_conditions) == 1:
            where_clause = where_conditions[0]
        elif len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}
        
        query_embedding = self.embedding_model.encode(natural_language_query).tolist()
        all_results = []
        
        for coll_name, collection in self.collections.items():
            # Se filtramos por zona, saltamos coleções que tendem a não ter zone_id para evitar incompatibilidade estrutural
            if zone_filter and coll_name in ["reports"]:
                continue
                
            try:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"]
                )
                
                if results and results["ids"] and results["ids"][0]:
                    for i in range(len(results["ids"][0])):
                        distance = results["distances"][0][i]
                        score = 1.0 - distance
                        all_results.append(RetrievalResult(
                            id=results["ids"][0][i],
                            content=results["documents"][0][i],
                            metadata=results["metadatas"][0][i],
                            distance=distance,
                            score=score
                        ))
            except Exception:
                continue
        
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]

    def synthesize_answer(self, query: str, retrieved_chunks: list[RetrievalResult]) -> dict[str, Any]:
        """Synthesize answer using Gemini based on retrieved chunks."""
        if not retrieved_chunks:
            return {
                "answer": "Não foi encontrada informação relevante nos registos históricos do sistema.",
                "citations": [],
                "confidence": 0.0,
                "insufficient_context": True
            }
        
        # Montar contexto para a LLM
        context_str = "\n".join([f"- [{c.id}] (Score: {c.score:.2f}): {c.content}" for c in retrieved_chunks])
        
        prompt = (
            "És o assistente inteligente de auditoria e monitorização de prateleiras da loja.\n"
            "Responde à questão do gestor de forma clara, curta e factual utilizando APENAS os registos históricos fornecidos abaixo.\n"
            "Se o contexto não contiver dados suficientes, informa explicitamente.\n\n"
            f"Contexto dos registos:\n{context_str}\n\n"
            f"Questão: {query}\n"
            "Resposta:"
        )
        
        try:
            response = self.genai_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2)
            )
            answer_text = response.text.strip()
        except Exception as e:
            answer_text = f"Erro na síntese da resposta via Gemini: {e}"
        
        citations = [
            {
                "id": c.id,
                "source": c.metadata.get("source", "unknown"),
                "zone_id": c.metadata.get("zone_id", "N/A"),
                "timestamp": c.metadata.get("timestamp", "N/A")
            }
            for c in retrieved_chunks
        ]
        
        mean_confidence = sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
        
        return {
            "answer": answer_text,
            "citations": citations,
            "confidence": round(mean_confidence, 3),
            "insufficient_context": "Não tenho" in answer_text or "insuficiente" in answer_text.lower()
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Memory CLI")
    parser.add_argument("--add-inspection", help="Add inspection JSON file")
    parser.add_argument("--add-tp1", action="store_true", help="Index TP1 trajectory data")
    parser.add_argument("--query", help="Natural language query")
    parser.add_argument("--zone", help="Filter by zone")
    parser.add_argument("--source", help="Filter by source (comma-separated)")
    
    args = parser.parse_args()
    rag = RAGMemory()
    
    if args.add_inspection:
        with open(args.add_inspection, encoding="utf-8") as f:
            data = json.load(f)
        # Utiliza o chunking granular obrigatório por problema
        chunk_ids = rag.add_inspection_per_issue(data)
        print(f"Inspeção adicionada. Gerados {len(chunk_ids)} chunks com sucesso.")
    
    if args.add_tp1:
        tp1_path = Path("./data/tp1_trajectory/metrics.json")
        if tp1_path.exists():
            with open(tp1_path, encoding="utf-8") as f:
                tp1_data = json.load(f)
            ids = rag.add_tp1_trajectory_data(tp1_data)
            print(f"Sucesso: Adicionados {len(ids)} chunks provenientes dos dados do TP1.")
        else:
            print(f"Erro: Ficheiro {tp1_path} não encontrado.")
    
    if args.query:
        source_filter = [SourceType(s.strip()) for s in args.source.split(",")] if args.source else None
        zone_filter = [args.zone.strip()] if args.zone else None
        results = rag.query(args.query, source_filter, zone_filter)
        answer = rag.synthesize_answer(args.query, results)
        print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()