#!/usr/bin/env python3
"""
Evaluation Harness for Retail Vision Intelligence System.
Tests 10+ images with ground truth, calculates mandatory metrics, and includes LLM-as-judge evaluation.
"""

from __future__ import annotations

import sys
import json
import time
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from src.shelf_inspector import ShelfInspector, Strategy  # noqa: E402
from src.rag_memory import RAGMemory  # noqa: E402


class GroundTruthIssue(BaseModel):
    """Ground truth issue for evaluation."""
    type: str
    location: str
    severity: str
    confidence_min: float = 0.0
    description_keywords: List[str] = []


class GroundTruthImage(BaseModel):
    """Ground truth for a single image."""
    image_path: str
    zone_id: str
    expected_issues: List[GroundTruthIssue]
    expected_status: str
    expected_fill_rate_range: Tuple[float, float]
    expected_products: List[str]


class EvaluationResult(BaseModel):
    """Result for a single image evaluation."""
    image_path: str
    strategy: str
    parsed_successfully: bool
    issue_detection_rate: float
    false_positive_rate: float
    severity_accuracy: float = 0.0
    json_parse_rate: float = 1.0
    hallucination_rate: float = 0.0
    latency_ms: float = 0.0
    detected_issues: List[Dict] = []
    missing_issues: List[Dict] = []
    extra_issues: List[Dict] = []
    cache_fallback: bool = False
    errors: List[str] = []


class MetricsSummary(BaseModel):
    """Aggregated metrics across all evaluations."""
    total_images: int
    total_evaluations: int
    avg_issue_detection_rate: float
    avg_false_positive_rate: float
    avg_severity_accuracy: float
    avg_hallucination_rate: float
    overall_json_parse_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    strategy_agreement_rate: float
    recall_at_3: float = 0.0
    rule_parse_rate: float = 0.0
    rule_correctness: float = 0.0
    ambiguity_detection_rate: float = 0.0
    rag_faithfulness: float = 0.0
    rag_answer_relevance: float = 0.0
    results: List[EvaluationResult] = []


class LLMJudgeResult(BaseModel):
    """Result from LLM-as-judge evaluation."""
    criterion_scores: Dict[str, Dict[str, Any]]
    overall_score: float
    summary: str


class Evaluator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str | None = None,
        ground_truth_path: str = "./data/ground_truth.json"
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured.")
        
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.ground_truth_path = Path(ground_truth_path)
        self.ground_truth: List[GroundTruthImage] = []
        self.results: List[EvaluationResult] = []
        
        self._load_ground_truth()
    
    def _load_ground_truth(self) -> None:
        """Load ground truth from JSON file."""
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")
        
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.ground_truth = [GroundTruthImage(**item) for item in data["images"]]
    
    def _compute_image_hash(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _match_issue(
        self,
        detected: Dict,
        expected: List[GroundTruthIssue]
    ) -> Optional[GroundTruthIssue]:
        """Check if a detected issue matches any expected issue."""
        det_type = detected.get("type", "")
        det_location = detected.get("location", "").lower()
        det_severity = detected.get("severity", "")
        
        for exp in expected:
            type_match = det_type == exp.type
            loc_match = exp.location.lower() in det_location or det_location in exp.location.lower()
            sev_match = det_severity == exp.severity
            
            if type_match and loc_match and sev_match:
                return exp
        return None
    
    def evaluate_image(
        self,
        image_path: str,
        strategy: Strategy,
        ground_truth: GroundTruthImage,
        use_cache: bool = True
    ) -> EvaluationResult:
        """Evaluate a single image with a specific strategy."""
        start_time = time.time()
        
        result = EvaluationResult(
            image_path=image_path,
            strategy=strategy.value,
            parsed_successfully=False,
            issue_detection_rate=0.0,
            false_positive_rate=0.0,
            detected_issues=[],
            missing_issues=[],
            extra_issues=[],
            errors=[]
        )
        
        try:
            inspector = ShelfInspector()
            inspection = inspector.inspect(
                image_path,
                zone_id=ground_truth.zone_id,
                strategy=strategy,
                use_cache=use_cache
            )
            
            result.parsed_successfully = True
            result.json_parse_rate = 1.0
            result.cache_fallback = inspection.cache_fallback
            
            detected_issues = [i.model_dump() for i in inspection.issues]
            result.detected_issues = detected_issues
            
            expected_issues = ground_truth.expected_issues
            
            matched = []
            missing = list(expected_issues)
            extra = list(detected_issues)
            
            for det in detected_issues:
                match = self._match_issue(det, expected_issues)
                if match:
                    matched.append(match)
                    if match in missing:
                        missing.remove(match)
                    if det in extra:
                        extra.remove(det)
            
            result.missing_issues = [m.model_dump() for m in missing]
            result.extra_issues = extra
            
            n_expected = len(expected_issues) if expected_issues else 1
            result.issue_detection_rate = len(matched) / n_expected if n_expected > 0 else 1.0
            result.false_positive_rate = len(extra) / max(len(detected_issues), 1) if detected_issues else 0.0
            
            sev_correct = 0
            sev_total = 0
            for det in detected_issues:
                match = self._match_issue(det, expected_issues)
                if match:
                    sev_total += 1
                    if det.get("severity") == match.severity:
                        sev_correct += 1
            result.severity_accuracy = sev_correct / sev_total if sev_total > 0 else 0.0
            
            hallucinated = 0
            known_products = set(p.lower() for p in ground_truth.expected_products)
            known_locations = set(iss.location.lower() for iss in expected_issues)
            
            for det in detected_issues:
                match = self._match_issue(det, expected_issues)
                if not match and det.get("description"):
                    desc_lower = det["description"].lower()
                    words = set(desc_lower.split())
                    unknown_claims = 0
                    for word in words:
                        if len(word) > 4 and word not in known_products and word not in known_locations:
                            if any(prod in word or word in prod for prod in known_products):
                                continue
                            unknown_claims += 1
                    if unknown_claims >= 2:
                        hallucinated += 1
            result.hallucination_rate = hallucinated / max(len(detected_issues), 1) if detected_issues else 0.0
            
        except json.JSONDecodeError as e:
            result.json_parse_rate = 0.0
            result.errors.append(f"JSON parse error: {str(e)}")
        except Exception as e:
            result.errors.append(str(e))
            result.cache_fallback = True
        
        end_time = time.time()
        result.latency_ms = (end_time - start_time) * 1000
        
        return result
    
    def evaluate_all(
        self,
        strategies: Optional[List[Strategy]] = None,
        use_cache: bool = True
    ) -> MetricsSummary:
        """Evaluate all ground truth images across all strategies."""
        if strategies is None:
            strategies = list(Strategy)
        
        self.results = []

        for gt_image in self.ground_truth:
            for strategy in strategies:
                try:
                    result = self.evaluate_image(
                        gt_image.image_path,
                        strategy,
                        gt_image,
                        use_cache=use_cache
                    )
                    self.results.append(result)
                except Exception as e:
                    self.results.append(EvaluationResult(
                        image_path=gt_image.image_path,
                        strategy=strategy.value,
                        parsed_successfully=False,
                        issue_detection_rate=0.0,
                        false_positive_rate=0.0,
                        detected_issues=[],
                        missing_issues=[],
                        extra_issues=[],
                        cache_fallback=False,
                        errors=[f"Evaluation aborted: {e}"],
                    ))

        return self._compute_summary()
    
    def _compute_summary(self) -> MetricsSummary:
        """Compute aggregated metrics."""
        if not self.results:
            return MetricsSummary(
                total_images=0,
                total_evaluations=0,
                avg_issue_detection_rate=0.0,
                avg_false_positive_rate=0.0,
                avg_severity_accuracy=0.0,
                avg_hallucination_rate=0.0,
                overall_json_parse_rate=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                strategy_agreement_rate=0.0,
                results=[]
            )
        
        latencies = sorted([r.latency_ms for r in self.results])
        n = len(latencies)
        p50 = latencies[int(n * 0.5)] if n > 0 else 0
        p95 = latencies[int(n * 0.95)] if n > 0 else 0
        
        detection_rates = [r.issue_detection_rate for r in self.results]
        fp_rates = [r.false_positive_rate for r in self.results]
        sev_accuracies = [r.severity_accuracy for r in self.results]
        hall_rates = [r.hallucination_rate for r in self.results]
        json_rates = [r.json_parse_rate for r in self.results]
        
        avg_detection = sum(detection_rates) / len(detection_rates) if detection_rates else 0
        avg_fp = sum(fp_rates) / len(fp_rates) if fp_rates else 0
        avg_sev = sum(sev_accuracies) / len(sev_accuracies) if sev_accuracies else 0
        avg_hall = sum(hall_rates) / len(hall_rates) if hall_rates else 0
        avg_json = sum(json_rates) / len(json_rates) if json_rates else 0
        
        agreement = self._compute_strategy_agreement()
        
        return MetricsSummary(
            total_images=len(self.ground_truth),
            total_evaluations=len(self.results),
            avg_issue_detection_rate=avg_detection,
            avg_false_positive_rate=avg_fp,
            avg_severity_accuracy=avg_sev,
            avg_hallucination_rate=avg_hall,
            overall_json_parse_rate=avg_json,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            strategy_agreement_rate=agreement,
            recall_at_3=0.0,
            rule_parse_rate=0.0,
            rule_correctness=0.0,
            ambiguity_detection_rate=0.0,
            rag_faithfulness=0.0,
            rag_answer_relevance=0.0,
            results=self.results
        )
    
    def _compute_strategy_agreement(self) -> float:
        """Compute agreement rate across strategies for same images."""
        if not self.ground_truth:
            return 0.0
        
        agreements = 0
        total = 0
        
        for gt in self.ground_truth:
            image_results = [r for r in self.results if r.image_path == gt.image_path]
            if len(image_results) < 2:
                continue
            
            statuses = []
            for r in image_results:
                if r.detected_issues:
                    statuses.append("has_issues")
                else:
                    statuses.append("no_issues")
            
            if len(set(statuses)) == 1:
                agreements += 1
            total += 1
        
        return agreements / total if total > 0 else 0.0
    
    def evaluate_rule_engine(self, rules_ground_truth_path: str = "./data/rule_ground_truth.json") -> Dict[str, float]:
        """Evaluate Rule Engine metrics: parse rate, correctness, ambiguity detection."""
        from src.rule_engine import RuleEngine
        
        if not Path(rules_ground_truth_path).exists():
            return {"rule_parse_rate": 0.0, "rule_correctness": 0.0, "ambiguity_detection_rate": 0.0}
        
        with open(rules_ground_truth_path, "r", encoding="utf-8") as f:
            rules_data = json.load(f)
        
        test_rules = rules_data.get("rules", [])
        if not test_rules:
            return {"rule_parse_rate": 0.0, "rule_correctness": 0.0, "ambiguity_detection_rate": 0.0}
        
        parsed = 0
        correct = 0
        ambiguity_detected = 0
        total = len(test_rules)
        
        for rule_gt in test_rules:
            nl_rule = rule_gt.get("natural_language", "")
            expected_valid = rule_gt.get("expected_valid", True)
            expected_ambiguous = rule_gt.get("expected_ambiguous", False)
            
            try:
                engine = RuleEngine()
                rule = engine.add_rule(nl_rule, auto_resolve=True)
                parsed += 1
                
                if rule.validation.is_valid == expected_valid:
                    correct += 1
                
                if expected_ambiguous and not rule.validation.is_valid:
                    ambiguity_detected += 1
                elif not expected_ambiguous and rule.validation.is_valid:
                    ambiguity_detected += 1
                    
            except Exception:
                if not expected_valid:
                    correct += 1
                if expected_ambiguous:
                    ambiguity_detected += 1
        
        return {
            "rule_parse_rate": parsed / total if total > 0 else 0.0,
            "rule_correctness": correct / total if total > 0 else 0.0,
            "ambiguity_detection_rate": ambiguity_detected / total if total > 0 else 0.0
        }
    
    def evaluate_rag_quality(self, queries: List[Dict[str, Any]], top_k: int = 3) -> Dict[str, float]:
        """Evaluate RAG Faithfulness and Answer Relevance using LLM-as-judge."""
        
        if not queries:
            return {"rag_faithfulness": 0.0, "rag_answer_relevance": 0.0}
        
        rag = RAGMemory()
        faithfulness_scores = []
        relevance_scores = []
        
        for query_item in queries:
            query_text = query_item.get("query", "")
            if not query_text:
                continue
            
            results = rag.query(query_text, top_k=top_k)
            answer = rag.synthesize_answer(query_text, results)
            
            if answer.get("insufficient_context", False):
                continue
            
            chunks_text = "\n\n".join([c.content for c in results])
            
            faithfulness_prompt = f"""Avalia a fidelidade (faithfulness) da seguinte resposta RAG aos chunks recuperados.
Resposta: {answer.get('answer', '')}
Chunks: {chunks_text}
Devolve JSON: {{"faithfulness": 0.0-1.0, "relevance": 0.0-1.0, "justification": "string"}}"""
            
            try:
                judge_response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=faithfulness_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    )
                )
                judge_data = json.loads(judge_response.text.strip())
                faithfulness_scores.append(judge_data.get("faithfulness", 0.0))
                relevance_scores.append(judge_data.get("relevance", 0.0))
            except Exception:
                faithfulness_scores.append(0.0)
                relevance_scores.append(0.0)
        
        return {
            "rag_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0,
            "rag_answer_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        }
    
    def evaluate_rag_recall(self, queries: list[dict[str, Any]], top_k: int = 3) -> float:
        """Evaluate RAG Recall@k using ground-truth expected IDs."""
        if not queries:
            return 0.0
        rag = RAGMemory()
        hits = 0
        total = 0
        for q in queries:
            query_text = q.get("query", "")
            expected_ids = set(q.get("expected_ids", []))
            if not query_text or not expected_ids:
                continue
            total += 1
            results = rag.query(query_text, top_k=top_k)
            retrieved_ids = set(r.id for r in results)
            if expected_ids & retrieved_ids:
                hits += 1
        return hits / total if total > 0 else 0.0

    def evaluate_chunking_comparison(self, test_inspections: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare hybrid vs per-issue chunking using Recall@3."""
        
        results = {
            "hybrid_recall": 0.0,
            "per_issue_recall": 0.0,
            "hybrid_chunk_count": 0,
            "per_issue_chunk_count": 0,
            "queries_evaluated": 0
        }
        
        hybrid_rag = RAGMemory()
        per_issue_rag = RAGMemory()
        
        hybrid_collection = hybrid_rag.collections["shelf_inspections"]
        per_issue_collection = per_issue_rag.collections["shelf_inspections"]
        
        hybrid_collection.delete(ids=hybrid_collection.get()["ids"])
        per_issue_collection.delete(ids=per_issue_collection.get()["ids"])
        
        for insp in test_inspections:
            try:
                hybrid_rag.add_inspection(insp)
            except Exception:
                pass
            try:
                per_issue_rag.add_inspection_per_issue(insp)
            except Exception:
                pass
        
        hybrid_ids = set(hybrid_collection.get()["ids"])
        per_issue_ids = set(per_issue_collection.get()["ids"])
        results["hybrid_chunk_count"] = len(hybrid_ids)
        results["per_issue_chunk_count"] = len(per_issue_ids)
        
        hybrid_hits = 0
        per_issue_hits = 0
        total_queries = 0
        
        for q in queries:
            query_text = q.get("query", "")
            expected_ids = set(q.get("expected_ids", []))
            if not query_text or not expected_ids:
                continue
            
            total_queries += 1
            
            hybrid_results = hybrid_rag.query(query_text, top_k=3)
            hybrid_retrieved = set(r.id for r in hybrid_results)
            if expected_ids & hybrid_retrieved:
                hybrid_hits += 1
            
            per_issue_results = per_issue_rag.query(query_text, top_k=3)
            per_issue_retrieved = set(r.id for r in per_issue_results)
            if expected_ids & per_issue_retrieved:
                per_issue_hits += 1
        
        results["queries_evaluated"] = total_queries
        results["hybrid_recall"] = hybrid_hits / total_queries if total_queries > 0 else 0.0
        results["per_issue_recall"] = per_issue_hits / total_queries if total_queries > 0 else 0.0
        
        return results
    
    def llm_as_judge(
        self,
        report_markdown: str,
        evaluation_type: str = "report",
        ground_truth: Optional[Dict] = None
    ) -> LLMJudgeResult:
        """Evaluate output using LLM-as-judge with prompt template."""
        
        prompt_template = Path(__file__).parent / "prompts" / "llm_judge.txt"
        if not prompt_template.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_template}")
        
        prompt = prompt_template.read_text(encoding="utf-8")
        
        gt_str = json.dumps(ground_truth, ensure_ascii=False, indent=2) if ground_truth else "N/A"
        
        filled_prompt = prompt.format(
            output_to_evaluate=report_markdown,
            evaluation_type=evaluation_type,
            ground_truth=gt_str
        )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=filled_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.strip()
        response_data = json.loads(response_text)
        
        return LLMJudgeResult(**response_data)
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save evaluation results to JSON."""
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = f"./output/evaluation_{timestamp}.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        summary = self._compute_summary()
        
        output_data = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "metrics": summary.model_dump(),
            "results": [r.model_dump() for r in self.results]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    
    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        summary = self._compute_summary()
        
        table = Table(title="Evaluation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Images", str(summary.total_images))
        table.add_row("Total Evaluations", str(summary.total_evaluations))
        table.add_row("Avg Issue Detection Rate", f"{summary.avg_issue_detection_rate:.1%}")
        table.add_row("Avg False Positive Rate", f"{summary.avg_false_positive_rate:.1%}")
        table.add_row("Overall JSON Parse Rate", f"{summary.overall_json_parse_rate:.1%}")
        table.add_row("Latency P50", f"{summary.latency_p50_ms:.0f} ms")
        table.add_row("Latency P95", f"{summary.latency_p95_ms:.0f} ms")
        table.add_row("Strategy Agreement Rate", f"{summary.strategy_agreement_rate:.1%}")
        table.add_row("Recall@3 (RAG)", f"{summary.recall_at_3:.1%}")
        
        console.print(table)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation Harness")
    parser.add_argument("--images-dir", help="Directory with test images (auto-discovers ground truth)")
    parser.add_argument("--ground-truth", default="./data/ground_truth.json", help="Ground truth JSON file (fallback)")
    parser.add_argument("--strategies", default="A,B,C", help="Comma-separated strategies (A,B,C)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--rag-queries", help="Path to RAG evaluation queries JSON")
    parser.add_argument("--rag-quality", help="Path to RAG quality queries JSON")
    parser.add_argument("--rules-gt", default="./data/rule_ground_truth.json", help="Rule engine ground truth JSON")
    parser.add_argument("--llm-judge", help="Path to report markdown for LLM-as-judge evaluation")
    
    args = parser.parse_args()
    
    ground_truth_path = args.ground_truth
    if args.images_dir:
        ground_truth_path = Path(args.images_dir) / "ground_truth.json"
    
    evaluator = Evaluator(ground_truth_path=str(ground_truth_path))
    
    strategy_list = [Strategy(s.strip().upper()) for s in args.strategies.split(",")]
    
    summary = evaluator.evaluate_all(strategies=strategy_list, use_cache=not args.no_cache)
    
    if args.rag_queries:
        with open(args.rag_queries, "r", encoding="utf-8") as f:
            rag_queries = json.load(f)
        recall = evaluator.evaluate_rag_recall(rag_queries)
        summary.recall_at_3 = recall
    
    if args.rag_quality:
        with open(args.rag_quality, "r", encoding="utf-8") as f:
            rag_quality_queries = json.load(f)
        rag_quality = evaluator.evaluate_rag_quality(rag_quality_queries)
        summary.rag_faithfulness = rag_quality["rag_faithfulness"]
        summary.rag_answer_relevance = rag_quality["rag_answer_relevance"]
    
    rule_metrics = evaluator.evaluate_rule_engine(args.rules_gt)
    summary.rule_parse_rate = rule_metrics["rule_parse_rate"]
    summary.rule_correctness = rule_metrics["rule_correctness"]
    summary.ambiguity_detection_rate = rule_metrics["ambiguity_detection_rate"]
    
    evaluator.print_summary()
    
    if args.llm_judge:
        with open(args.llm_judge, "r", encoding="utf-8") as f:
            report_text = f.read()
        judge_result = evaluator.llm_as_judge(report_text)
        console.print(f"\n[bold]LLM-as-Judge Score:[/bold] {judge_result.overall_score:.1f}/10")
        console.print(f"[bold]Summary:[/bold] {judge_result.summary}")
    
    output_path = evaluator.save_results(args.output)
    console.print(f"\n[green]Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    main()
