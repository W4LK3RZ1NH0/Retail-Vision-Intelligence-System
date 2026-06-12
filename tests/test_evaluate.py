import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from evaluate import (  # noqa: E402
    Evaluator,
    EvaluationResult,
    MetricsSummary,
    GroundTruthIssue,
    GroundTruthImage,
    LLMJudgeResult,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

FAKE_GT = [
    GroundTruthImage(
        image_path="img_a.jpg",
        zone_id="Z_S1",
        expected_issues=[
            GroundTruthIssue(type="empty_shelf", location="top", severity="high"),
            GroundTruthIssue(type="wrong_product", location="middle", severity="medium"),
        ],
        expected_status="critical",
        expected_fill_rate_range=(0.3, 0.6),
        expected_products=["leite"],
    ),
    GroundTruthImage(
        image_path="img_b.jpg",
        zone_id="Z_S2",
        expected_issues=[
            GroundTruthIssue(type="misaligned", location="bottom", severity="low"),
        ],
        expected_status="warning",
        expected_fill_rate_range=(0.6, 0.9),
        expected_products=["pão", "queijo"],
    ),
]


def make_evaluator(tmp_path, gt_images=None):
    """Build an Evaluator with a real ground-truth file on disk."""
    gt_images = gt_images or FAKE_GT
    gt_file = tmp_path / "ground_truth.json"
    gt_file.write_text(
        json.dumps({"images": [img.model_dump() for img in gt_images]}),
        encoding="utf-8",
    )
    with patch("evaluate.genai") as mock_genai, \
         patch.object(LLMJudgeResult, "model_validate", return_value=LLMJudgeResult(
             criterion_scores={}, overall_score=7.5, summary="OK"
         )):
        mock_genai.Client.return_value = MagicMock()
        evaluator = Evaluator(ground_truth_path=str(gt_file))
    evaluator.ground_truth = gt_images
    return evaluator


# ── MetricsSummary (early-return path) ────────────────────────────────────────

class TestMetricsSummaryEmpty:
    def test_empty_results_has_all_fields(self):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.ground_truth = []
        evaluator.results = []

        summary = evaluator._compute_summary()

        assert summary.total_images == 0
        assert summary.total_evaluations == 0
        assert summary.avg_issue_detection_rate == 0.0
        assert summary.avg_false_positive_rate == 0.0
        assert summary.avg_severity_accuracy == 0.0
        assert summary.avg_hallucination_rate == 0.0
        assert summary.overall_json_parse_rate == 0.0
        assert summary.latency_p50_ms == 0.0
        assert summary.latency_p95_ms == 0.0
        assert summary.strategy_agreement_rate == 0.0
        assert summary.recall_at_3 == 0.0
        assert summary.rule_parse_rate == 0.0
        assert summary.rule_correctness == 0.0
        assert summary.ambiguity_detection_rate == 0.0
        assert summary.rag_faithfulness == 0.0
        assert summary.rag_answer_relevance == 0.0
        assert summary.results == []

        # Must not raise on model_dump
        data = summary.model_dump()
        assert isinstance(data, dict)


# ── _compute_summary with real data ──────────────────────────────────────────

class TestMetricsSummaryPopulated:
    def _build_evaluator(self, tmp_path):
        return make_evaluator(tmp_path)

    def test_aggregation(self, tmp_path):
        evaluator = self._build_evaluator(tmp_path)
        evaluator.results = [
            EvaluationResult(
                image_path="img_a.jpg",
                strategy="A",
                parsed_successfully=True,
                issue_detection_rate=1.0,
                false_positive_rate=0.0,
                severity_accuracy=1.0,
                json_parse_rate=1.0,
                hallucination_rate=0.0,
                latency_ms=1000.0,
                detected_issues=[{"type": "empty_shelf", "location": "top", "severity": "high"}],
                missing_issues=[],
                extra_issues=[],
                cache_fallback=False,
                errors=[],
            ),
            EvaluationResult(
                image_path="img_a.jpg",
                strategy="B",
                parsed_successfully=True,
                issue_detection_rate=0.5,
                false_positive_rate=0.2,
                severity_accuracy=0.5,
                json_parse_rate=1.0,
                hallucination_rate=0.1,
                latency_ms=2000.0,
                detected_issues=[{"type": "empty_shelf", "location": "top", "severity": "high"}],
                missing_issues=[],
                extra_issues=[],
                cache_fallback=False,
                errors=[],
            ),
            EvaluationResult(
                image_path="img_b.jpg",
                strategy="A",
                parsed_successfully=True,
                issue_detection_rate=0.0,
                false_positive_rate=0.0,
                severity_accuracy=0.0,
                json_parse_rate=0.0,  # simula falha de parse
                hallucination_rate=0.0,
                latency_ms=500.0,
                detected_issues=[],
                missing_issues=[],
                extra_issues=[],
                cache_fallback=False,
                errors=["JSON parse error"],
            ),
        ]

        summary = evaluator._compute_summary()

        assert summary.total_images == 2
        assert summary.total_evaluations == 3
        assert summary.avg_issue_detection_rate == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert summary.avg_false_positive_rate == pytest.approx((0.0 + 0.2 + 0.0) / 3)
        assert summary.avg_severity_accuracy == pytest.approx((1.0 + 0.5 + 0.0) / 3)
        assert summary.avg_hallucination_rate == pytest.approx((0.0 + 0.1 + 0.0) / 3)
        assert summary.overall_json_parse_rate == pytest.approx((1.0 + 1.0 + 0.0) / 3)
        assert summary.latency_p50_ms == 1000.0
        assert summary.latency_p95_ms == 2000.0
        # img_a: both strategies detect issues -> agreed; img_b: both no issues -> agreed
        assert summary.strategy_agreement_rate == 1.0


# ── _match_issue ──────────────────────────────────────────────────────────────

class TestMatchIssue:
    def _mk_evaluator(self):
        return Evaluator.__new__(Evaluator)

    def test_exact_match(self):
        ev = self._mk_evaluator()
        expected = [GroundTruthIssue(type="t", location="loc", severity="s")]
        assert ev._match_issue({"type": "t", "location": "loc", "severity": "s"}, expected) is not None

    def test_partial_location_match(self):
        ev = self._mk_evaluator()
        expected = [GroundTruthIssue(type="t", location="top shelf left", severity="s")]
        assert ev._match_issue({"type": "t", "location": "top shelf", "severity": "s"}, expected) is not None

    def test_no_match_different_type(self):
        ev = self._mk_evaluator()
        expected = [GroundTruthIssue(type="t", location="loc", severity="s")]
        assert ev._match_issue({"type": "other", "location": "loc", "severity": "s"}, expected) is None

    def test_no_match_different_severity(self):
        ev = self._mk_evaluator()
        expected = [GroundTruthIssue(type="t", location="loc", severity="high")]
        assert ev._match_issue({"type": "t", "location": "loc", "severity": "low"}, expected) is None


# ── evaluate_image failure path ───────────────────────────────────────────────

class TestEvaluateImageFailure:
    def test_missing_image_file(self, tmp_path):
        evaluator = make_evaluator(tmp_path)
        fake_strategy = type("StrategyEnum", (), {"value": "A"})()
        result = evaluator.evaluate_image(
            str(tmp_path / "does_not_exist.jpg"),
            fake_strategy,
            FAKE_GT[0],
            use_cache=False,
        )
        assert result.parsed_successfully is False
        assert result.latency_ms > 0
        assert len(result.errors) > 0
        assert result.strategy == "A"


# ── LLM judge template resolution ─────────────────────────────────────────────

class TestLLMJudgeTemplate:
    def test_template_exists(self):
        p = Path(__file__).parent.parent / "prompts" / "llm_judge.txt"
        assert p.exists(), f"Missing prompt: {p}"
        content = p.read_text(encoding="utf-8")
        assert "output_to_evaluate" in content
        assert "evaluation_type" in content
        assert "ground_truth" in content


# ── Rule engine ground-truth missing ─────────────────────────────────────────

class TestRuleEngineFallback:
    def test_missing_rule_gt_returns_zeros(self):
        evaluator = make_evaluator(Path(tempfile.mkdtemp()))
        result = evaluator.evaluate_rule_engine("./nonexistent_rule_gt.json")
        assert result == {
            "rule_parse_rate": 0.0,
            "rule_correctness": 0.0,
            "ambiguity_detection_rate": 0.0,
        }


# ── RAG quality empty-query guard ─────────────────────────────────────────────

class TestRAGQualityGuards:
    def test_empty_queries_returns_zeros(self):
        evaluator = make_evaluator(Path(tempfile.mkdtemp()))
        result = evaluator.evaluate_rag_quality([])
        assert result == {"rag_faithfulness": 0.0, "rag_answer_relevance": 0.0}


# ── save_results writes valid JSON ────────────────────────────────────────────

class TestSaveResults:
    def test_save_and_load(self, tmp_path):
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.ground_truth = []
        evaluator.results = []

        out = evaluator.save_results(str(tmp_path / "out.json"))
        assert os.path.exists(out)

        with open(out, "r", encoding="utf-8") as f:
            raw = json.load(f)

        assert "timestamp" in raw
        assert "Z" in raw["timestamp"]
        assert "metrics" in raw
        assert "results" in raw
        assert raw["results"] == []

        # MetricsSummary deserialises cleanly
        ms = MetricsSummary(**raw["metrics"])
        assert ms.total_images == 0


# ── CLI top-level entry point ─────────────────────────────────────────────────

class TestCLI:
    def test_argparse_recognises_all_flags(self):
        """Verify all expected CLI flags are registered (sanity check)."""
        import evaluate

        src = Path(evaluate.__file__).read_text(encoding="utf-8")
        for flag in [
            "--images-dir",
            "--ground-truth",
            "--strategies",
            "--no-cache",
            "--output",
            "--rag-queries",
            "--rag-quality",
            "--rules-gt",
            "--llm-judge",
        ]:
            assert flag in src, f"Missing CLI flag: {flag}"
