#!/usr/bin/env python3
"""
Report Generator - Creates comprehensive Markdown reports from inspection sessions.
Integrates shelf inspections, rule violations, RAG historical context, and TP1 trajectory data.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

from src.rag_memory import RetrievalResult  # noqa: E402
from src.shelf_inspector import InspectionResult  # noqa: E402


class ReportData(BaseModel):
    """Input data for report generation."""
    session_id: str
    timestamp: str
    inspections: list[InspectionResult]
    triggered_rules: list[dict[str, Any]]
    rag_context: list[RetrievalResult] = []

    include_llm_judge: bool = False
    llm_judge_result: dict[str, Any] | None = None
    tp1_data: dict[str, Any] | None = None

    class Config:
        arbitrary_types_allowed = True


class ReportGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY não configurado. Define-o no ficheiro .env.")
        
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=self.api_key)
        self.temperature = temperature
        
        prompts_dir = Path(__file__).parent.parent / "prompts"
        self.prompt_path = prompts_dir / "report_generator.txt"
        if self.prompt_path.exists():
            self.report_prompt = self.prompt_path.read_text(encoding="utf-8")
        else:
            # Fallback robusto caso o ficheiro não exista temporariamente
            self.report_prompt = (
                "Gera um relatório de auditoria de retalho com base nos seguintes dados:\n\n"
                "Dados da Sessão:\n{session_data}\n\n"
                "Contexto Histórico RAG:\n{rag_context}\n\n"
                "Métricas de Tráfego TP1:\n{tp1_data}\n"
            )
    
    def _prepare_session_data(self, report_data: ReportData) -> str:
        """Format session data for the prompt template."""
        lines = []
        lines.append(f"SESSÃO: {report_data.session_id}")
        lines.append(f"TIMESTAMP: {report_data.timestamp}")
        lines.append(f"ZONAS INSPECIONADAS: {', '.join(set(i.zone_id for i in report_data.inspections))}")
        lines.append("")
        
        for insp in report_data.inspections:
            lines.append(f"--- INSPEÇÃO {insp.inspection_id} ---")
            lines.append(f"Zona: {insp.zone_id}")
            lines.append(f"Timestamp: {insp.timestamp}")
            # Suporta tanto Enums do Pydantic (.value) como strings nativas
            status_val = insp.overall_status.value if hasattr(insp.overall_status, 'value') else insp.overall_status
            lines.append(f"Status Geral: {status_val}")
            lines.append(f"Fill Rate: {insp.shelf_fill_rate:.1%}")
            lines.append(f"Produtos Detectados: {', '.join(insp.products_detected) if insp.products_detected else 'N/A'}")
            lines.append(f"Estratégia: {insp.strategy_used}")
            lines.append(f"Cache Fallback: {insp.cache_fallback}")
            lines.append("")
            
            if insp.issues:
                lines.append("ISSUES:")
                for issue in insp.issues:
                    type_val = issue.type.value if hasattr(issue.type, 'value') else issue.type
                    severity_val = issue.severity.value if hasattr(issue.severity, 'value') else issue.severity
                    lines.append(f"  - {issue.issue_id}: {type_val} | {severity_val} | {issue.location} | conf={issue.confidence:.2f} | área={issue.affected_area_pct:.1f}%")
                    lines.append(f"    Descrição: {issue.description}")
            else:
                lines.append("ISSUES: Nenhum")
            lines.append("")
            
            reasoning = insp.model_reasoning or "N/A"
            lines.append(f"RACIOCÍNIO: {reasoning[:500]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def _prepare_rag_context(self, rag_results: list[RetrievalResult]) -> str:
        """Format RAG context for the prompt."""
        if not rag_results:
            return "Nenhum contexto histórico recuperado."
        
        lines = []
        for i, chunk in enumerate(rag_results, 1):
            lines.append(f"CHUNK {i} (score: {chunk.score:.3f}):")
            lines.append(f"  ID: {chunk.id}")
            lines.append(f"  Fonte: {chunk.metadata.get('source', 'N/A')}")
            lines.append(f"  Zona: {chunk.metadata.get('zone_id', 'N/A')}")
            
            timestamp = chunk.metadata.get('timestamp', 'N/A')
            date_str = timestamp[:10] if timestamp else 'N/A'
            lines.append(f"  Data: {date_str}")
            lines.append(f"  Status: {chunk.metadata.get('overall_status', 'N/A')}")
            
            try:
                fr = float(chunk.metadata.get('fill_rate', 0.0))
                lines.append(f"  Fill Rate: {fr:.1%}")
            except (ValueError, TypeError):
                lines.append(f"  Fill Rate: {chunk.metadata.get('fill_rate', 'N/A')}")
                
            lines.append(f"  Conteúdo: {chunk.content}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _prepare_tp1_data(self, tp1_data: dict[str, Any] | None, inspected_zones: list[str], inspections: list[InspectionResult]) -> str:
        """Format relevant TP1 trajectory data with correlation analysis."""
        if not tp1_data:
            return "Dados de trajetória TP1 não disponíveis."
        
        lines = []
        lines.append("=== DADOS TP1 TRAJECTÓRIA E CORRELAÇÃO ===")
        
        zona_data = tp1_data.get("zona", {})
        trafego_data = tp1_data.get("trafego", {})
        anomalia_data = tp1_data.get("anomalia", {})
        
        # Extract inspection hours safely
        inspection_hours = []
        for insp in inspections:
            if insp.timestamp and len(insp.timestamp) >= 13:
                try:
                    hour = int(insp.timestamp[11:13])
                    inspection_hours.append(hour)
                except ValueError:
                    pass
        
        # Traffic per inspected zone
        lines.append("\n--- TRÁFEGO POR ZONA INSPECIONADA ---")
        zone_issues = {}
        for insp in inspections:
            zone = insp.zone_id
            high_severity_count = 0
            if insp.issues:
                for issue in insp.issues:
                    sev = issue.severity.value if hasattr(issue.severity, 'value') else issue.severity
                    if sev in ["high", "critical", "CRITICAL", "HIGH"]:
                        high_severity_count += 1
            zone_issues[zone] = high_severity_count
        
        for zone in inspected_zones:
            traffic = zona_data.get("traffic_per_zone", {}).get(zone, 0)
            dwell = zona_data.get("avg_dwell_time_per_zone_seconds", {}).get(zone, 0.0)
            stop_rate = zona_data.get("stopping_rate_per_zone", {}).get(zone, 0.0)
            critical_issues = zone_issues.get(zone, 0)
            
            lines.append(f"  {zone}: {traffic} visitantes | {dwell:.0f}s dwell | {stop_rate:.1%} stop rate | {critical_issues} issues críticos/altos")
            
            if traffic > 1000 and critical_issues > 0:
                lines.append(f"    ⚠ CORRELAÇÃO: Alta afluência ({traffic}) com issues críticos - risco iminente de rotura de stock e perda de vendas.")
        
        # Hourly traffic vs inspection timing
        lines.append("\n--- TRÁFEGO HORÁRIO vs HORA DE INSPEÇÃO ---")
        hourly = trafego_data.get("unique_visitors_per_hour", {})
        for hour in sorted(hourly.keys(), key=int):
            marker = " [HORA DE INSPEÇÃO]" if int(hour) in inspection_hours else ""
            lines.append(f"  {hour}h: {hourly[hour]}{marker}")
        
        # Anomalies in inspected zones
        lines.append("\n--- ANOMALIAS DE TRÁFEGO NAS ZONAS INSPECIONADAS ---")
        anomalies = anomalia_data.get("anomalies_detected", [])
        relevant_anomalies = [a for a in anomalies if a.get("zone_id") in inspected_zones]
        
        if relevant_anomalies:
            for a in relevant_anomalies[:10]:
                lines.append(f"  {a['zone_id']} às {a['hour']}h: {a['direction']} (σ={a['sigma_distance']:.1f}) | baseline={a['baseline_mean']:.1f} vs real={a['day_7_value']}")
                if a["direction"] == "ABOVE_EXPECTED":
                    lines.append("    → O tráfego invulgarmente alto sugere picos de procura ou estrangulamentos na reposição.")
                elif a["direction"] == "BELOW_EXPECTED":
                    lines.append("    → Queda abrupta de tráfego pode indicar que a prateleira está vazia ou sem sinalização.")
        else:
            lines.append("  Nenhuma anomalia de tráfego reportada para as zonas analisadas.")
        
        # Overall traffic context
        lines.append("\n--- CONTEXTO GERAL DE TRÁFEGO ---")
        weekly = trafego_data.get("weekly_affluence_distribution", {})
        today = datetime.now().strftime("%A")
        today_traffic = weekly.get(today, 0)
        avg_traffic = sum(weekly.values()) / len(weekly) if weekly else 0
        
        lines.append(f"  Tráfego hoje ({today}): {today_traffic} (média semanal base: {avg_traffic:.0f})")
        if avg_traffic > 0 and today_traffic > avg_traffic * 1.2:
            pct_increase = (today_traffic / avg_traffic - 1) * 100
            lines.append(f"  ⚠ Alerta: Dia de elevadíssima afluência (+{pct_increase:.0f}%) - stress operacional acrescido nas prateleiras.")
        
        return "\n".join(lines)
    
    def generate_report(self, report_data: ReportData) -> str:
        """Generate the markdown report using Gemini."""
        session_data = self._prepare_session_data(report_data)
        rag_context = self._prepare_rag_context(report_data.rag_context)
        inspected_zones = list(set(i.zone_id for i in report_data.inspections))
        tp1_data = self._prepare_tp1_data(report_data.tp1_data, inspected_zones, report_data.inspections)
        
        # Substituição limpa baseada em dicionário para mitigar KeyErrors causados por chavetas soltas no ficheiro de texto
        prompt = self.report_prompt
        for key, val in [("{session_data}", session_data), ("{rag_context}", rag_context), ("{tp1_data}", tp1_data)]:
            if key in prompt:
                prompt = prompt.replace(key, val)
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=4096
            )
        )
        
        report_md = response.text.strip()
        
        if report_data.llm_judge_result:
            report_md = self._append_llm_judge_section(report_md, report_data.llm_judge_result)
            
        return report_md
    
    def _append_llm_judge_section(self, report_markdown: str, llm_judge_result: dict[str, Any] | None) -> str:
        if not llm_judge_result:
            return report_markdown
        
        section = "\n\n## 7. Avaliação Qualitativa (LLM-as-Judge)\n\n"
        scores = llm_judge_result.get("criterion_scores", {})
        
        table_rows = []
        for criterion, data in scores.items():
            score = data.get("score", 0)
            justification = data.get("justification", "")
            table_rows.append(f"| {criterion} | {score:.1f}/10 | {justification} |")
        
        overall = llm_judge_result.get("overall_score", 0)
        summary = llm_judge_result.get("summary", "")
        
        section += f"**Score Global de Auditoria: {overall:.1f}/10**\n\n"
        section += "| Critério | Score | Justificação |\n"
        section += "|----------|-------|---------------|\n"
        section += "\n".join(table_rows)
        section += f"\n\n**Resumo do Juiz:** {summary}\n"
        
        return report_markdown + section
    
    def save_report(self, report_markdown: str, output_path: str | None = None) -> str:
        """Save report to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"./output/report_{timestamp}.md"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_markdown)
        
        return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Report Generator CLI")
    parser.add_argument("--session", required=True, help="Session JSON file with inspection data")
    parser.add_argument("--rag-query", help="Natural language query for RAG context")
    parser.add_argument("--output", help="Output markdown file")
    parser.add_argument("--tp1", action="store_true", help="Include TP1 trajectory data")
    
    args = parser.parse_args()
    
    with open(args.session, encoding="utf-8") as f:
        session_data = json.load(f)
    
    # Desserialização em conformidade estrita com Pydantic v2
    inspections = []
    for i in session_data.get("inspections", []):
        try:
            inspections.append(InspectionResult.model_validate(i))
        except ValidationError as ve:
            print(f"Aviso: Falha ao validar inspeção individual. Ignorando item. Detalhes: {ve}")
            
    triggered_rules = session_data.get("triggered_rules", [])
    
    rag_context = []
    if args.rag_query:
        from src.rag_memory import RAGMemory, SourceType
        try:
            rag = RAGMemory()
            source_filter = [SourceType.SHELF_INSPECTION, SourceType.RULE_VIOLATION]
            rag_results = rag.query(args.rag_query, source_filter=source_filter, top_k=5)
            # Garante compatibilidade forçando a re-validação ou atribuição direta mapeada se necessário
            rag_context = rag_results
        except Exception as e:
            print(f"Aviso: Não foi possível carregar contexto do RAG Memory: {e}")
    
    tp1_data = None
    if args.tp1:
        tp1_path = Path("./data/tp1_trajectory/metrics.json")
        if tp1_path.exists():
            with open(tp1_path, encoding="utf-8") as f:
                tp1_data = json.load(f)
        else:
            print(f"Aviso: Ficheiro de métricas TP1 ({tp1_path}) não localizado.")
    
    try:
        report_data = ReportData(
            session_id=session_data.get("session_id", f"SESS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            inspections=inspections,
            triggered_rules=triggered_rules,
            rag_context=rag_context,
            tp1_data=tp1_data
        )
        
        generator = ReportGenerator()
        report = generator.generate_report(report_data)
        output_path = generator.save_report(report, args.output)
        print(f"Report saved successfully to: {output_path}")
        
    except ValidationError as e:
        print(f"Erro crítico de validação ao construir os dados do relatório: {e}")


if __name__ == "__main__":
    main()