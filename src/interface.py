#!/usr/bin/env python3
"""
CLI Interface for Retail Vision Intelligence System.
Implements all modes specified in Section 8 of the specification.
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
from datetime import datetime, timezone

# --- Configuração ---
load_dotenv()
app = typer.Typer(name="retail-vision", help="Retail Vision Intelligence System")
console = Console()

# --- Sub-apps ---
inspector_app = typer.Typer(help="Shelf inspection commands")
rules_app = typer.Typer(help="Rule management commands")
query_app = typer.Typer(help="RAG query commands")
index_app = typer.Typer(help="Index management commands")
report_app = typer.Typer(help="Report generation commands")

app.add_typer(inspector_app, name="inspect")
app.add_typer(rules_app, name="rules")
app.add_typer(query_app, name="query")
app.add_typer(index_app, name="index")
app.add_typer(report_app, name="report")

# --- Helper ---

def _safe_print_json(data):
    try:
        console.print_json(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:
        console.print(f"[red]Error rendering output: {exc}[/red]")


# --- Modo de inspeção ---

@inspector_app.command("single")
def inspect_single(
    image: str = typer.Argument(..., help="Path to shelf image"),
    zone: str = typer.Option("Z_S1", "--zone", "-z"),
    output: Optional[str] = typer.Option(None, "--output", "-o")
):
    """Inspect a single shelf image."""
    from src.shelf_inspector import ShelfInspector
    try:
        inspector = ShelfInspector()
        result = inspector.inspect(image, zone)
        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
            console.print(f"[green]Result saved to: {output}[/green]")
        else:
            _safe_print_json(result.model_dump())
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@inspector_app.command("all")
def inspect_all(
    images_dir: str = typer.Argument(..., help="Directory with images"),
    zone: str = typer.Option("Z_S1", "--zone", "-z"),
    output: Optional[str] = typer.Option(None, "--output", "-o")
):
    """Inspect all images in a directory."""
    from src.shelf_inspector import ShelfInspector
    try:
        inspector = ShelfInspector()
        image_paths = sorted(Path(images_dir).glob("*.jpg"))
        if not image_paths:
            console.print("[yellow]No .jpg images found in directory.[/yellow]")
            raise typer.Exit(0)
        results = []
        for img in image_paths:
            try:
                r = inspector.inspect(str(img), zone)
                results.append(r.model_dump())
            except Exception as e:
                console.print(f"[yellow]Skipped {img.name}: {e}[/yellow]")
        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Batch results saved to: {output}[/green]")
        else:
            console.print(f"[green]Inspected {len(results)}/{len(image_paths)} images[/green]")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# --- Modo de definição de regras ---

@rules_app.command("add")
def rules_add(
    natural_language: str = typer.Argument(..., help="Rule in natural language")
):
    """Add a new rule from natural language."""
    from src.rule_engine import RuleEngine
    try:
        engine = RuleEngine()
        rule = engine.add_rule(natural_language)
        console.print(f"[green]Rule {rule.rule_id} added.[/green]")
        if not rule.validation.is_valid:
            console.print("[yellow]Ambiguities detected:[/yellow]")
            for a in rule.validation.ambiguities:
                console.print(f"  - {a}")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@rules_app.command("list")
def rules_list():
    """List all rules."""
    from src.rule_engine import RuleEngine
    try:
        engine = RuleEngine()
        rules = engine.list_rules()
        if not rules:
            console.print("[yellow]No rules found.[/yellow]")
            return
        table = Table(title="Rules")
        table.add_column("ID", style="cyan")
        table.add_column("Status", style="green")
        for rule in rules:
            table.add_row(rule.rule_id, "VALID" if rule.validation.is_valid else "INVALID")
        console.print(table)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@rules_app.command("delete")
def rules_delete(rule_id: str = typer.Argument(..., help="Rule ID to delete")):
    """Delete a rule by ID."""
    from src.rule_engine import RuleEngine
    try:
        engine = RuleEngine()
        if engine.delete_rule(rule_id):
            console.print(f"[green]Rule {rule_id} deleted.[/green]")
        else:
            console.print(f"[yellow]Rule {rule_id} not found.[/yellow]")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@rules_app.command("test")
def rules_test(
    rule_id: str = typer.Argument(..., help="Rule ID to test"),
    image: str = typer.Argument(..., help="Image path")
):
    """Test a rule against an image."""
    from src.rule_engine import RuleEngine
    from src.shelf_inspector import ShelfInspector
    try:
        inspector = ShelfInspector()
        inspection = inspector.inspect(image)
        engine = RuleEngine()
        result = engine.test_rule(rule_id, inspection)
        _safe_print_json(result)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# --- Modo de consulta histórica ---

@query_app.command("history")
def query_history(query: str = typer.Argument(..., help="Natural language history query")):
    """Query historical RAG memory."""
    from src.rag_memory import RAGMemory
    try:
        rag = RAGMemory()
        results = rag.query(query, top_k=3)
        answer = rag.synthesize_answer(query, results)
        _safe_print_json(answer)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@query_app.command("compare")
def query_compare(
    zone_a: str = typer.Argument(..., help="First zone"),
    zone_b: str = typer.Argument(..., help="Second zone"),
    period: str = typer.Option("last 7 days", "--period", "-p")
):
    """Compare two zones over a period."""
    from src.rag_memory import RAGMemory
    try:
        rag = RAGMemory()
        q = f"Compare zones {zone_a} and {zone_b} over {period}"
        results = rag.query(q, zone_filter=[zone_a, zone_b], top_k=5)
        answer = rag.synthesize_answer(q, results)
        _safe_print_json(answer)
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# --- Modo de relatório ---

@report_app.command("generate")
def report_generate(
    session: str = typer.Argument(..., help="Session JSON file with inspection data"),
    zone: Optional[str] = typer.Option(None, "--zone", "-z"),
    period: str = typer.Option("last 14 days", "--period", "-p"),
    output: Optional[str] = typer.Option(None, "--output", "-o")
):
    """Generate an inspection report."""
    from src.report_generator import ReportGenerator, ReportData
    from src.rag_memory import RAGMemory, SourceType
    try:
        session_path = Path(session)
        if not session_path.exists():
            console.print(f"[red]Session file not found: {session}[/red]")
            raise typer.Exit(1)
        with open(session_path, encoding="utf-8") as f:
            session_data = json.load(f)
        inspections = []
        for i in session_data.get("inspections", []):
            try:
                from src.shelf_inspector import InspectionResult
                inspections.append(InspectionResult.model_validate(i))
            except Exception:
                pass
        rag_context = []
        if zone:
            rag = RAGMemory()
            rag_results = rag.query(
                f"Last inspections in {zone} over {period}",
                source_filter=[SourceType.SHELF_INSPECTION],
                top_k=5,
            )
            rag_context = rag_results
        report_data = ReportData(
            session_id=session_data.get("session_id", f"SESS_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"),
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            inspections=inspections,
            triggered_rules=session_data.get("triggered_rules", []),
            rag_context=rag_context,
        )
        generator = ReportGenerator()
        md = generator.generate_report(report_data)
        out = generator.save_report(md, output)
        console.print(f"[green]Report saved to: {out}[/green]")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@report_app.command("zone")
def report_zone(
    zone: str = typer.Argument(..., help="Zone to generate report for"),
    period: str = typer.Option("last 14 days", "--period", "-p"),
    output: Optional[str] = typer.Option(None, "--output", "-o")
):
    """Generate a report for a specific zone."""
    from src.rag_memory import RAGMemory, SourceType
    from src.report_generator import ReportGenerator, ReportData
    try:
        rag = RAGMemory()
        q = f"Inspection report for zone {zone} over {period}"
        rag_results = rag.query(
            q,
            source_filter=[SourceType.SHELF_INSPECTION, SourceType.RULE_VIOLATION],
            zone_filter=[zone],
            top_k=5,
        )
        report_data = ReportData(
            session_id=f"ZONE_REPORT_{zone}",
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            inspections=[],
            triggered_rules=[],
            rag_context=rag_results,
        )
        generator = ReportGenerator()
        md = generator.generate_report(report_data)
        out = generator.save_report(md, output)
        console.print(f"[green]Zone report saved to: {out}[/green]")
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# --- Utilitários ---

@app.command()
def version():
    """Show version information."""
    console.print("[bold]Retail Vision Intelligence System[/bold] v1.0.0")


def main():
    app()


if __name__ == "__main__":
    main()
