#!/usr/bin/env python3
"""
Rule Engine - Natural language rule processing for retail shelf inspection.
Converts NL to JSON configurations, detects ambiguities, and executes rule conditions.
"""

import json
import os
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class LocationFilter(str, Enum):
    BOTTOM = "bottom"
    MIDDLE = "middle"
    TOP = "top"
    ANY = "any"


class IssueType(str, Enum):
    EMPTY_SHELF = "empty_shelf"
    WRONG_PRODUCT = "wrong_product"
    DAMAGED = "damaged"
    MISALIGNED = "misaligned"
    LABEL_MISSING = "label_missing"
    OTHER = "other"


class SeverityThreshold(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RuleCondition(BaseModel):
    zone_filter: list[str] | None = None
    time_filter: dict[str, int] | None = None
    issue_types: list[IssueType] | None = None
    severity_threshold: SeverityThreshold | None = None
    fill_rate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    location_filter: LocationFilter | None = None


class RuleAction(BaseModel):
    alert_level: AlertLevel
    notification_message: str


class RuleValidation(BaseModel):
    is_valid: bool
    ambiguities: list[str] = []
    assumptions: list[str] = []


class Rule(BaseModel):
    rule_id: str
    created_at: str
    natural_language: str
    description: str
    conditions: RuleCondition
    action: RuleAction
    validation: RuleValidation


class AmbiguityDetail(BaseModel):
    aspect: str
    description: str
    clarification_question: str
    suggested_options: list[str] | None = None


class AmbiguityAnalysis(BaseModel):
    ambiguities: list[AmbiguityDetail] = []
    missing_required_fields: list[str] = []
    can_proceed_with_assumptions: bool = False


class RuleEngine:
    def __init__(
        self,
        api_key: str | None = None,
        rules_dir: str = "./data/rules",
        model_name: str | None = None,
        temperature: float = 0.0
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY not configured. Set it in .env file.")
        
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.temperature = temperature
        self.client = genai.Client(api_key=self.api_key)
        
        self.rules_dir = Path(rules_dir)
        self.rules_dir.mkdir(parents=True, exist_ok=True)
        self.rules_file = self.rules_dir / "rules.json"
        
        prompts_dir = Path(__file__).parent.parent / "prompts"
        self.rule_parser_prompt = (prompts_dir / "rule_parser.txt").read_text(encoding="utf-8")
        self.rule_ambiguity_prompt = (prompts_dir / "rule_ambiguity.txt").read_text(encoding="utf-8")
        
        self.rules: dict[str, Rule] = {}
        self._load_rules()
        self._rule_counter = len(self.rules)
    
    def _load_rules(self) -> None:
        if self.rules_file.exists():
            try:
                with open(self.rules_file, encoding="utf-8") as f:
                    data = json.load(f)
                for rule_data in data:
                    rule = Rule(**rule_data)
                    self.rules[rule.rule_id] = rule
            except Exception as e:
                print(f"Warning: Failed to load rules: {e}", file=sys.stderr)
    
    def _save_rules(self) -> None:
        rules_list = [rule.model_dump() for rule in self.rules.values()]
        with open(self.rules_file, "w", encoding="utf-8") as f:
            json.dump(rules_list, f, ensure_ascii=False, indent=2)
    
    def _generate_rule_id(self) -> str:
        self._rule_counter += 1
        return f"RULE_{self._rule_counter:03d}"
    
    def _call_gemini(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                response_mime_type="application/json"
            )
        )
        return response.text
    
    def parse_rule(self, natural_language: str) -> Rule:
        prompt = self.rule_parser_prompt.replace("{natural_language}", natural_language)
        response_text = self._call_gemini(prompt)
        rule_data = json.loads(response_text)
        
        if "rule_id" not in rule_data or not rule_data["rule_id"]:
            rule_data["rule_id"] = self._generate_rule_id()
        if "created_at" not in rule_data:
            rule_data["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        return Rule(**rule_data)
    
    def detect_ambiguities(self, natural_language: str) -> AmbiguityAnalysis:
        prompt = self.rule_ambiguity_prompt.replace("{natural_language}", natural_language)
        response_text = self._call_gemini(prompt)
        analysis_data = json.loads(response_text)
        return AmbiguityAnalysis(**analysis_data)
    
    def add_rule(self, natural_language: str, auto_resolve: bool = False) -> Rule:
        ambiguity_analysis = self.detect_ambiguities(natural_language)
        
        if ambiguity_analysis.ambiguities and not auto_resolve:
            rule = self.parse_rule(natural_language)
            rule.validation.is_valid = False
            rule.validation.ambiguities = [a.description for a in ambiguity_analysis.ambiguities]
            return rule
        
        rule = self.parse_rule(natural_language)
        rule.validation.is_valid = True
        
        if ambiguity_analysis.ambiguities:
            rule.validation.ambiguities = [a.description for a in ambiguity_analysis.ambiguities]
            rule.validation.assumptions = [
                f"Assumed: {a.aspect} = {a.suggested_options[0] if a.suggested_options else 'default'}"
                for a in ambiguity_analysis.ambiguities
            ]
        
        self.rules[rule.rule_id] = rule
        self._save_rules()
        return rule
    
    def list_rules(self) -> list[Rule]:
        return list(self.rules.values())
    
    def get_rule(self, rule_id: str) -> Rule | None:
        return self.rules.get(rule_id)
    
    def delete_rule(self, rule_id: str) -> bool:
        if rule_id in self.rules:
            del self.rules[rule_id]
            self._save_rules()
            return True
        return False
    
    def _evaluate_condition(self, condition: RuleCondition, inspection) -> bool:
        if condition.zone_filter and inspection.zone_id not in condition.zone_filter:
            return False
        
        if condition.time_filter:
            inspection_hour = int(inspection.timestamp[11:13])
            if inspection_hour < condition.time_filter.get("hours_start", 0):
                return False
            if inspection_hour > condition.time_filter.get("hours_end", 23):
                return False
        
        if condition.issue_types:
            issue_types_in_inspection = {issue.type.value for issue in inspection.issues}
            required_types = {t.value for t in condition.issue_types}
            if not required_types.intersection(issue_types_in_inspection):
                return False
        
        if condition.severity_threshold:
            severity_order = {"low": 1, "medium": 2, "high": 3}
            threshold_level = severity_order[condition.severity_threshold.value]
            max_severity = max(
                (severity_order.get(issue.severity.value, 0) for issue in inspection.issues),
                default=0
            )
            if max_severity < threshold_level:
                return False
        
        if condition.fill_rate_threshold is not None and inspection.shelf_fill_rate >= condition.fill_rate_threshold:
            return False
        
        if condition.location_filter and condition.location_filter != LocationFilter.ANY:
            locations = {issue.location.lower() for issue in inspection.issues}
            filter_loc = condition.location_filter.value
            if not any(filter_loc in loc for loc in locations):
                return False
        
        return True
    
    def execute_rules(self, inspection) -> list[dict[str, Any]]:
        triggered = []
        
        for rule in self.rules.values():
            if not rule.validation.is_valid:
                continue
            
            if self._evaluate_condition(rule.conditions, inspection):
                notification = rule.action.notification_message.format(
                    zone_id=inspection.zone_id,
                    issue_type=", ".join({i.type.value for i in inspection.issues}) if inspection.issues else "N/A",
                    severity=", ".join({i.severity.value for i in inspection.issues}) if inspection.issues else "N/A",
                    fill_rate=f"{inspection.shelf_fill_rate:.0%}",
                    location=", ".join({i.location for i in inspection.issues}) if inspection.issues else "any",
                    inspection_id=inspection.inspection_id,
                    timestamp=inspection.timestamp
                )
                
                triggered.append({
                    "rule_id": rule.rule_id,
                    "rule_description": rule.description,
                    "alert_level": rule.action.alert_level.value,
                    "notification": notification,
                    "inspection_id": inspection.inspection_id,
                    "zone_id": inspection.zone_id,
                    "matched_issues": [i.model_dump() for i in inspection.issues]
                })
        
        self._log_execution(inspection, triggered)
        return triggered
    
    def _log_execution(self, inspection, triggered: list[dict]) -> None:
        log_dir = Path("./data/log")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "rule_execution.log"
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "inspection_id": inspection.inspection_id,
            "zone_id": inspection.zone_id,
            "rules_checked": len(self.rules),
            "rules_triggered": len(triggered),
            "triggered_rules": triggered
        }
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    
    def test_rule(self, rule_id: str, inspection) -> dict[str, Any]:
        rule = self.get_rule(rule_id)
        if not rule:
            return {"error": f"Rule {rule_id} not found"}
        
        if not rule.validation.is_valid:
            return {"error": f"Rule {rule_id} is not valid", "ambiguities": rule.validation.ambiguities}
        
        matches = self._evaluate_condition(rule.conditions, inspection)
        
        result = {
            "rule_id": rule_id,
            "matches": matches,
            "conditions": rule.conditions.model_dump()
        }
        
        if matches:
            notification = rule.action.notification_message.format(
                zone_id=inspection.zone_id,
                issue_type=", ".join({i.type.value for i in inspection.issues}) if inspection.issues else "N/A",
                severity=", ".join({i.severity.value for i in inspection.issues}) if inspection.issues else "N/A",
                fill_rate=f"{inspection.shelf_fill_rate:.0%}",
                location=", ".join({i.location for i in inspection.issues}) if inspection.issues else "any",
                inspection_id=inspection.inspection_id,
                timestamp=inspection.timestamp
            )
            result["notification"] = notification
            result["alert_level"] = rule.action.alert_level.value
        
        return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rule Engine CLI")
    parser.add_argument("--add", help="Add rule from natural language")
    parser.add_argument("--list", action="store_true", help="List all rules")
    parser.add_argument("--delete", help="Delete rule by ID")
    parser.add_argument("--test", help="Test rule against inspection JSON file")
    parser.add_argument("--inspection", help="Inspection JSON file for testing")
    
    args = parser.parse_args()
    engine = RuleEngine()
    
    if args.add:
        # Lógica de Clarificação Interativa da Secção 5.4 do Enunciado
        print(f"A analisar possíveis ambiguidades para a regra: '{args.add}'...")
        analysis = engine.detect_ambiguities(args.add)
        
        refined_nl = args.add
        if analysis.ambiguities:
            print("\n⚠ Foram detetadas ambiguidades na regra fornecida:")
            for a in analysis.ambiguities:
                print(f"\n-> Aspeto: {a.aspect}")
                print(f"   {a.clarification_question}")
                if a.suggested_options:
                    print(f"   Opções recomendadas: {', '.join(a.suggested_options)}")
                
                user_choice = input("   A tua resposta (ou enter para assumir padrão): ").strip()
                if user_choice:
                    refined_nl += f" ({a.aspect}: {user_choice})"
            
            print("\nA processar regra refinada final...")
            rule = engine.add_rule(refined_nl, auto_resolve=True)
        else:
            rule = engine.add_rule(refined_nl)
            
        print("\nRegra Registada com Sucesso:")
        print(json.dumps(rule.model_dump(), ensure_ascii=False, indent=2))
        
    elif args.list:
        rules = engine.list_rules()
        for r in rules:
            status = "VALID" if r.validation.is_valid else "INVALID"
            print(f"{r.rule_id}: {r.natural_language} [{status}]")
    elif args.delete:
        if engine.delete_rule(args.delete):
            print(f"Rule {args.delete} deleted")
        else:
            print(f"Rule {args.delete} not found")
    elif args.test and args.inspection:
        with open(args.inspection, encoding="utf-8") as f:
            inspection_data = json.load(f)
        from src.shelf_inspector import InspectionResult
        inspection = InspectionResult(**inspection_data)
        result = engine.test_rule(args.test, inspection)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()