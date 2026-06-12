#!/usr/bin/env python3
"""
Shelf Inspector - Visual analysis of retail shelf images using Google Gemini 1.5 Flash API
with MD5 caching, rate limiting (15 req/min), and three prompting strategies.
"""

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field, field_validator

load_dotenv()

logger = logging.getLogger(__name__)


class IssueType(str, Enum):
    EMPTY_SHELF = "empty_shelf"
    WRONG_PRODUCT = "wrong_product"
    DAMAGED = "damaged"
    MISALIGNED = "misaligned"
    LABEL_MISSING = "label_missing"
    OTHER = "other"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OverallStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class Issue(BaseModel):
    issue_id: str
    type: IssueType
    location: str
    severity: Severity
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    affected_area_pct: float = Field(ge=0.0, le=100.0)


class InspectionResult(BaseModel):
    inspection_id: str
    timestamp: str
    image_path: str
    zone_id: str
    overall_status: OverallStatus
    issues: list[Issue] = []
    shelf_fill_rate: float = Field(ge=0.0, le=1.0)
    products_detected: list[str] = []
    model_reasoning: str
    cache_fallback: bool = False
    strategy_used: str = "A"

    @field_validator("overall_status", mode="before")
    @classmethod
    def validate_status(cls, v):
        if isinstance(v, str):
            return OverallStatus(v.lower())
        return v

    @field_validator("shelf_fill_rate", mode="before")
    @classmethod
    def normalize_fill_rate(cls, v):
        if isinstance(v, (int, float)) and v > 1.0:
            return float(v) / 100.0
        return float(v)


class Strategy(Enum):
    A = "A"
    B = "B"
    # Adicionado suporte implícito caso venham em minúsculas
    C = "C"


@dataclass
class RateLimiter:
    """Sliding window rate limiter for 15 requests per minute."""
    max_requests: int = 15
    window_seconds: int = 60
    timestamps: list[float] = field(default_factory=list)

    def wait_if_needed(self) -> None:
        now = time.time()
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.window_seconds]

        if len(self.timestamps) >= self.max_requests:
            oldest = self.timestamps[0]
            wait_time = self.window_seconds - (now - oldest) + 0.1
            if wait_time > 0:
                time.sleep(wait_time)
            now = time.time()
            self.timestamps = [ts for ts in self.timestamps if now - ts < self.window_seconds]

        self.timestamps.append(time.time())


class TemporaryRateLimitError(Exception):
    """Erro temporário de rate limit (429) que justifica retry com backoff."""
    pass


class GeminiQuotaError(Exception):
    """Exceção customizada para identificar falha de quota esgotada permanentemente."""
    pass


class ShelfInspector:
    quota_dead: bool = False
    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: str = "./cache/shelf_inspector",
        model_name: str | None = None,
        temperature: float = 0.0
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY not configured. Set it in .env file.")

        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=self.api_key)
        self.temperature = temperature

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.rate_limiter = RateLimiter()

        self.prompts = {}
        prompts_dir = Path(__file__).parent.parent / "prompts"
        for strategy in Strategy:
            prompt_file = prompts_dir / f"shelf_inspector_{strategy.value.lower()}.txt"
            if prompt_file.exists():
                self.prompts[strategy] = prompt_file.read_text(encoding="utf-8")
            else:
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    def _compute_image_hash(self, image_path: str) -> str:
        """Compute MD5 hash of image file."""
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_cache_path(self, image_hash: str, strategy_value: str) -> Path:
        # Crucial: Separar o ficheiro de cache por estratégia!
        return self.cache_dir / f"{image_hash}_strat_{strategy_value.lower()}.json"

    def _load_from_cache(self, image_hash: str, strategy_value: str) -> InspectionResult | None:
        cache_path = self._get_cache_path(image_hash, strategy_value)
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as f:
                    data = json.load(f)
                return InspectionResult(**data)
            except Exception:
                return None
        return None

    def _save_to_cache(self, image_hash: str, strategy_value: str, result: InspectionResult) -> None:
        cache_path = self._get_cache_path(image_hash, strategy_value)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    def _is_quota_exhausted(self, error: Exception) -> bool:
        """Verifica se o erro diz respeito a esgotamento de quota ou rate limit (não retentável)."""
        error_str = str(error).lower()
        return (
            "quota" in error_str
            or "resource_exhausted" in error_str
            or "429" in error_str
            or "too many requests" in error_str
            or "rate limit" in error_str
        )

    def _log_api_error(self, image_hash: str, error: Exception) -> None:
        """Log API errors to file."""
        log_dir = Path("./data/log")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "api_errors.log"

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "image_hash": image_hash,
            "error_message": str(error)[:200]
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _call_gemini_raw(self, prompt: str, image_path: str) -> str:
        if not ShelfInspector.quota_dead:
            self.rate_limiter.wait_if_needed()
        image = Image.open(image_path)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    response_mime_type="application/json"
                )
            )
            return str(response.text)
        except Exception as e:
            if self._is_temporary_rate_limit(e):
                raise
            if self._is_quota_exhausted(e):
                raise GeminiQuotaError(f"Quota diária esgotada: {e}") from None
            raise

    def _is_temporary_rate_limit(self, error: Exception) -> bool:
        """Detecta 429 temporário (ainda dentro do rate limit, não quota esgotada)."""
        error_str = str(error).lower()
        if "429" not in error_str and "too many requests" not in error_str:
            return False
        if self._is_quota_exhausted(error):
            return False
        return True

    def _build_fallback_result(
        self,
        image_path: str,
        zone_id: str,
        strategy: Strategy,
        reason: str
    ) -> InspectionResult:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        inspection_id = f"INS_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{self._compute_image_hash(image_path)[:6]}"
        return InspectionResult(
            inspection_id=inspection_id,
            timestamp=timestamp,
            image_path=image_path,
            zone_id=zone_id,
            overall_status=OverallStatus.WARNING,
            issues=[],
            shelf_fill_rate=0.0,
            products_detected=[],
            model_reasoning=f"Fallback: {reason}",
            cache_fallback=False,
            strategy_used=strategy.value,
        )

    def _call_gemini(self, prompt: str, image_path: str) -> str:
        """Wrapper para intercetar falhas de quota imediata e saltar retentativas inúteis."""
        if ShelfInspector.quota_dead:
            raise GeminiQuotaError("Quota diária esgotada (modo cache-only)")
        try:
            return self._call_gemini_raw(prompt, image_path)
        except GeminiQuotaError:
            ShelfInspector.quota_dead = True
            raise
        except Exception as e:
            if self._is_quota_exhausted(e):
                ShelfInspector.quota_dead = True
                raise GeminiQuotaError(f"Quota Diária Completamente Esgotada: {e}") from None
            raise

    def inspect(
        self,
        image_path: str,
        zone_id: str = "Z_S1",
        strategy: Strategy = Strategy.B,
        use_cache: bool = True
    ) -> InspectionResult:
        if ShelfInspector.quota_dead:
            if use_cache:
                image_hash = self._compute_image_hash(image_path)
                cached = self._load_from_cache(image_hash, strategy.value)
                if cached:
                    cached.cache_fallback = True
                    print(f"[CACHE-ONLY] Quota esgotada — retornado cache para {Path(image_path).name}", file=sys.stderr)
                    return cached
            fb = self._build_fallback_result(image_path, zone_id, strategy, "Quota diária esgotada (modo cache-only)")
            print(f"[FALLBACK] Quota esgotada — resultado fallback para {Path(image_path).name}", file=sys.stderr)
            return fb

        image_hash = self._compute_image_hash(image_path)

        if use_cache:
            cached = self._load_from_cache(image_hash, strategy.value)
            if cached:
                cached.cache_fallback = True
                return cached

        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        inspection_id = f"INS_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{image_hash[:6]}"

        prompt = self.prompts.get(strategy)
        if not prompt:
            raise ValueError(f"Strategy {strategy.value} not available")

        try:
            response_text = self._call_gemini(prompt, image_path)
            result_data = json.loads(response_text)

            # Defesa contra outputs desalinhados ou falta de model_reasoning estruturado
            if "model_reasoning" not in result_data:
                result_data["model_reasoning"] = "Aviso: O modelo não gerou a cadeia de raciocínio no JSON."

            result_data["inspection_id"] = inspection_id
            result_data["timestamp"] = timestamp
            result_data["image_path"] = image_path
            result_data["zone_id"] = zone_id
            result_data["strategy_used"] = strategy.value
            result_data["cache_fallback"] = False

            result = InspectionResult(**result_data)

            if use_cache:
                self._save_to_cache(image_hash, strategy.value, result)

            return result

        except Exception as e:
            self._log_api_error(image_hash, e)
            logger.warning(f"Erro na API Gemini para {image_hash}: {e}. Tentando cache fallback.")

            if use_cache:
                cached = self._load_from_cache(image_hash, strategy.value)
                if cached:
                    cached.cache_fallback = True
                    print(f"AVISO: Falha na API ({e}). Recuperado fallback da cache local.", file=sys.stderr)
                    return cached

            # Quota esgotada: retorna resultado de fallback em vez de crashar
            error_str = str(e).lower()
            is_quota = (
                "quota" in error_str
                or "resource_exhausted" in error_str
                or "429" in error_str
                or "too many requests" in error_str
                or "rate limit" in error_str
            )
            if is_quota:
                ShelfInspector.quota_dead = True
                return self._build_fallback_result(image_path, zone_id, strategy, f"Quota diária esgotada: {e}")

            raise RuntimeError(f"A inspeção falhou e não existe cache disponível para esta estratégia: {e}") from None

    def inspect_all_strategies(
        self,
        image_path: str,
        zone_id: str = "Z_S1",
        use_cache: bool = True
    ) -> dict:
        """Run inspection with all three strategies for comparison."""
        results = {}
        for strategy in Strategy:
            try:
                results[strategy] = self.inspect(image_path, zone_id, strategy, use_cache)
            except Exception as e:
                print(f"Strategy {strategy.value} failed: {e}", file=sys.stderr)
                results[strategy] = None
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shelf Inspector CLI")
    parser.add_argument("--image", required=True, help="Path to shelf image")
    parser.add_argument("--zone", default="Z_S1", help="Zone ID")
    parser.add_argument("--strategy", choices=["A", "B", "C"], default="B", help="Prompting strategy")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--all-strategies", action="store_true", help="Run all strategies")

    args = parser.parse_args()

    try:
        inspector = ShelfInspector()
    except Exception as e:
        print(f"Erro de Inicialização: {e}", file=sys.stderr)
        sys.exit(1)

    if args.all_strategies:
        results = inspector.inspect_all_strategies(args.image, args.zone, not args.no_cache)
        output = {s.value: r.model_dump() if r else None for s, r in results.items()}
    else:
        # Suporta tanto maiúsculas como minúsculas passadas por argumento
        strategy = Strategy(args.strategy.upper())
        try:
            result = inspector.inspect(args.image, args.zone, strategy, not args.no_cache)
            output = result.model_dump()
        except Exception as e:
            print(f"Erro Crítico: {e}", file=sys.stderr)
            sys.exit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()