from __future__ import annotations

import json
import re
from typing import Any, Protocol

from litellm import completion

from trading_firm.config import AppConfig


class LLMClient(Protocol):
    backend: str

    def generate_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> dict[str, Any]: ...


class LLMError(RuntimeError):
    pass


def build_llm_client(config: AppConfig, force_mock: bool = False) -> LLMClient:
    if force_mock or config.llm_backend == "mock":
        return MockLLMClient(config)
    return LiteLLMClient(config)


class LiteLLMClient:
    backend = "litellm"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        candidates = _dedupe(
            [model or self.config.default_model, *self.config.fallback_models]
        )

        strict_system_prompt = (
            f"{system_prompt}\n\n"
            "Return a single valid JSON object only. Do not include markdown fences or commentary."
        )

        for candidate in candidates:
            try:
                response = completion(
                    model=candidate,
                    messages=[
                        {"role": "system", "content": strict_system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.config.temperature,
                    timeout=self.config.request_timeout,
                )
                content = response.choices[0].message.content or "{}"
                return _extract_json_object(content)
            except Exception as exc:  # pragma: no cover - network/provider dependent
                last_error = exc

        raise LLMError(f"Unable to complete task '{task_name}': {last_error}")


class MockLLMClient:
    backend = "mock"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        lower_prompt = user_prompt.lower()
        bullish_hits = sum(
            lower_prompt.count(token)
            for token in [
                "bullish",
                "uptrend",
                "growth",
                "beat",
                "positive",
                "strength",
            ]
        )
        bearish_hits = sum(
            lower_prompt.count(token)
            for token in [
                "bearish",
                "downtrend",
                "miss",
                "negative",
                "weakness",
                "risk",
            ]
        )
        score = max(-60, min(60, (bullish_hits - bearish_hits) * 8))
        stance = "bullish" if score > 8 else "bearish" if score < -8 else "neutral"

        if task_name.endswith("_analyst"):
            agent = task_name.replace("_analyst", "")
            return {
                "title": f"{agent.title()} view on current setup",
                "stance": stance,
                "score": score,
                "confidence": 58,
                "summary": f"Mock {agent} analysis based on the collected market context.",
                "key_points": [
                    "Recent data provides enough context to form a preliminary view.",
                    "Signals are mixed, so conviction should stay moderate.",
                    "New data could materially change the setup.",
                ],
                "risks": [
                    "Market regime can shift quickly around macro events.",
                    "Headline risk may override the current signal stack.",
                ],
                "data_points": [
                    "The framework is running in mock mode.",
                    "Outputs are placeholders until a real LLM backend is enabled.",
                ],
            }

        if task_name in {"bullish_researcher", "bearish_researcher"}:
            debate_stance = "bullish" if task_name.startswith("bullish") else "bearish"
            return {
                "stance": debate_stance,
                "conviction": 55,
                "thesis": f"Mock {debate_stance} memo built from the analyst packet.",
                "supporting_points": [
                    "The debate process consolidates multiple analyst viewpoints.",
                    "Positioning should reflect the balance of evidence.",
                ],
                "rebuttals": [
                    "Counter-signals keep the case from becoming high conviction.",
                    "Execution should wait for confirmation if timing is unclear.",
                ],
                "watch_items": [
                    "Monitor earnings, guidance, and macro headlines.",
                    "Reassess if volatility expands materially.",
                ],
            }

        if task_name == "trader":
            action = "BUY" if score > 12 else "SELL" if score < -12 else "HOLD"
            position_pct = 8.0 if action != "HOLD" else 0.0
            return {
                "action": action,
                "confidence": 56,
                "position_pct": position_pct,
                "time_horizon": "swing-trade, 1-4 weeks",
                "thesis": "Mock trade plan synthesized from analysts and debate memos.",
                "entry_rationale": [
                    "Enter only if the signal still aligns with the latest session data.",
                    "Avoid chasing extended moves without confirmation.",
                ],
                "exit_conditions": [
                    "Exit on thesis break or a clean stop-out.",
                    "Reduce if macro news invalidates the setup.",
                ],
                "primary_risks": [
                    "Unexpected news flow can change the setup immediately.",
                    "Mock mode does not validate reasoning quality.",
                ],
            }

        if task_name == "risk_manager":
            return {
                "approved": True,
                "max_position_pct": 6.0,
                "stop_loss_pct": 4.5,
                "take_profit_pct": 9.0,
                "risk_notes": [
                    "Cap size because conviction is moderate.",
                    "Use a defined stop before entering the trade.",
                ],
                "required_changes": [
                    "Confirm liquidity and spreads before sending the order.",
                ],
            }

        if task_name == "portfolio_manager":
            return {
                "approved": True,
                "final_action": "BUY" if score > 0 else "SELL" if score < 0 else "HOLD",
                "final_position_pct": 6.0 if score else 0.0,
                "rationale": "Mock portfolio approval with risk-capped sizing.",
                "conditions": [
                    "Proceed only if the risk manager constraints are met.",
                    "Refresh the analyst packet before executing large positions.",
                ],
            }

        raise LLMError(f"Unsupported mock task: {task_name}")


def _dedupe(models: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for model in models:
        if model and model not in seen:
            seen.add(model)
            unique.append(model)
    return unique


def _extract_json_object(content: str) -> dict[str, Any]:
    raw = content.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise LLMError(f"Model did not return JSON: {content[:200]}")

    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"Unable to parse model JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise LLMError("Expected a top-level JSON object from model output")
    return parsed
