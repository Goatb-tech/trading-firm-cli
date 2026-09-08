from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from trading_firm.config import AppConfig
from trading_firm.llm import LLMClient
from trading_firm.models import (
    AnalystReport,
    ContextSnapshot,
    DebateMemo,
    PortfolioDecision,
    RiskReview,
    TradeDecision,
)


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [str(value).strip()]


def _bounded_int(value: Any, default: int = 50) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def _bounded_score(value: Any, default: int = 0) -> int:
    try:
        return max(-100, min(100, int(value)))
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class AnalystAgent:
    agent: str
    focus: str

    def run(
        self, snapshot: ContextSnapshot, llm: LLMClient, config: AppConfig
    ) -> AnalystReport:
        context = self._context(snapshot)
        payload = llm.generate_json(
            task_name=f"{self.agent}_analyst",
            model=config.model_for(self.agent),
            system_prompt=(
                f"You are the {self.agent} analyst at a systematic trading firm. {self.focus} "
                "Use only the provided context. Be concrete, skeptical, and decision-oriented."
            ),
            user_prompt=(
                "Analyze the provided market packet and return JSON with keys: "
                "title, stance, score, confidence, summary, key_points, risks, data_points. "
                "Use stance as bullish, bearish, or neutral. Use score from -100 to 100 and confidence from 0 to 100.\n\n"
                f"Context:\n{_compact_json(context)}"
            ),
        )
        return AnalystReport(
            agent=self.agent,
            title=str(payload.get("title") or f"{self.agent.title()} Analyst Report"),
            stance=str(payload.get("stance") or "neutral").lower(),
            score=_bounded_score(payload.get("score"), 0),
            confidence=_bounded_int(payload.get("confidence"), 50),
            summary=str(payload.get("summary") or "No summary provided."),
            key_points=_list(payload.get("key_points")),
            risks=_list(payload.get("risks")),
            data_points=_list(payload.get("data_points")),
            raw=payload,
        )

    def _context(self, snapshot: ContextSnapshot) -> dict[str, Any]:
        if self.agent == "fundamentals":
            return {
                "symbol": snapshot.symbol,
                "company_name": snapshot.company_name,
                "fundamentals": snapshot.fundamentals,
                "price_summary": snapshot.price_summary,
                "company_news": snapshot.company_news[:3],
                "data_warnings": snapshot.data_warnings,
            }
        if self.agent == "sentiment":
            return {
                "symbol": snapshot.symbol,
                "company_name": snapshot.company_name,
                "social_sentiment": snapshot.social_sentiment,
                "company_news_headlines": [
                    item.get("title") for item in snapshot.company_news[:5]
                ],
                "data_warnings": snapshot.data_warnings,
            }
        if self.agent == "technical":
            return {
                "symbol": snapshot.symbol,
                "company_name": snapshot.company_name,
                "technicals": snapshot.technicals,
                "price_summary": snapshot.price_summary,
                "macro_snapshot": snapshot.macro_snapshot,
            }
        return {
            "symbol": snapshot.symbol,
            "company_name": snapshot.company_name,
            "macro_news": snapshot.macro_news,
            "macro_snapshot": snapshot.macro_snapshot,
            "company_news": snapshot.company_news,
            "data_warnings": snapshot.data_warnings,
        }


@dataclass(slots=True)
class ResearchAgent:
    agent: str
    stance: str

    def run(
        self,
        snapshot: ContextSnapshot,
        reports: list[AnalystReport],
        llm: LLMClient,
        config: AppConfig,
    ) -> DebateMemo:
        payload = llm.generate_json(
            task_name=f"{self.agent}_researcher",
            model=config.model_for(self.agent),
            system_prompt=(
                f"You are the {self.stance} researcher in an internal investment debate. "
                f"Argue the {self.stance} side as strongly as the evidence allows without inventing facts."
            ),
            user_prompt=(
                "Review the analyst packet and return JSON with keys: stance, conviction, thesis, supporting_points, rebuttals, watch_items. "
                "Conviction must be 0 to 100.\n\n"
                f"Market context:\n{_compact_json({'symbol': snapshot.symbol, 'company_name': snapshot.company_name, 'price_summary': snapshot.price_summary, 'technicals': snapshot.technicals})}\n\n"
                f"Analyst packet:\n{_compact_json([report.to_dict() for report in reports])}"
            ),
        )
        return DebateMemo(
            agent=self.agent,
            stance=str(payload.get("stance") or self.stance).lower(),
            conviction=_bounded_int(payload.get("conviction"), 50),
            thesis=str(payload.get("thesis") or "No thesis provided."),
            supporting_points=_list(payload.get("supporting_points")),
            rebuttals=_list(payload.get("rebuttals")),
            watch_items=_list(payload.get("watch_items")),
            raw=payload,
        )


class TraderAgent:
    def run(
        self,
        snapshot: ContextSnapshot,
        reports: list[AnalystReport],
        memos: list[DebateMemo],
        llm: LLMClient,
        config: AppConfig,
    ) -> TradeDecision:
        payload = llm.generate_json(
            task_name="trader",
            model=config.model_for("trader"),
            system_prompt=(
                "You are the lead trader. Synthesize the analyst reports and debate memos into a specific trade plan. "
                "Prefer HOLD when the evidence is weak or contradictory."
            ),
            user_prompt=(
                "Return JSON with keys: action, confidence, position_pct, time_horizon, thesis, entry_rationale, exit_conditions, primary_risks. "
                "Action must be BUY, SELL, or HOLD. Confidence is 0 to 100. Position_pct is percent of portfolio.\n\n"
                f"Market context:\n{_compact_json(snapshot.to_dict())}\n\n"
                f"Analyst reports:\n{_compact_json([report.to_dict() for report in reports])}\n\n"
                f"Debate memos:\n{_compact_json([memo.to_dict() for memo in memos])}"
            ),
        )
        action = str(payload.get("action") or "HOLD").upper()
        if action not in {"BUY", "SELL", "HOLD"}:
            action = "HOLD"
        return TradeDecision(
            action=action,
            confidence=_bounded_int(payload.get("confidence"), 50),
            position_pct=max(0.0, _float(payload.get("position_pct"), 0.0)),
            time_horizon=str(payload.get("time_horizon") or "unspecified"),
            thesis=str(payload.get("thesis") or "No thesis provided."),
            entry_rationale=_list(payload.get("entry_rationale")),
            exit_conditions=_list(payload.get("exit_conditions")),
            primary_risks=_list(payload.get("primary_risks")),
            raw=payload,
        )


class RiskManagerAgent:
    def run(
        self,
        snapshot: ContextSnapshot,
        trade: TradeDecision,
        llm: LLMClient,
        config: AppConfig,
    ) -> RiskReview:
        payload = llm.generate_json(
            task_name="risk_manager",
            model=config.model_for("risk"),
            system_prompt=(
                "You are the risk manager. Stress-test the proposed trade, cap the size appropriately, and reject the setup if controls are weak."
            ),
            user_prompt=(
                "Return JSON with keys: approved, max_position_pct, stop_loss_pct, take_profit_pct, risk_notes, required_changes.\n\n"
                f"Market context:\n{_compact_json({'price_summary': snapshot.price_summary, 'technicals': snapshot.technicals, 'macro_snapshot': snapshot.macro_snapshot})}\n\n"
                f"Trade plan:\n{_compact_json(trade.to_dict())}"
            ),
        )
        return RiskReview(
            approved=bool(payload.get("approved", False)),
            max_position_pct=max(0.0, _float(payload.get("max_position_pct"), 0.0)),
            stop_loss_pct=max(0.0, _float(payload.get("stop_loss_pct"), 0.0)),
            take_profit_pct=max(0.0, _float(payload.get("take_profit_pct"), 0.0)),
            risk_notes=_list(payload.get("risk_notes")),
            required_changes=_list(payload.get("required_changes")),
            raw=payload,
        )


class PortfolioManagerAgent:
    def run(
        self,
        snapshot: ContextSnapshot,
        reports: list[AnalystReport],
        trade: TradeDecision,
        risk: RiskReview,
        llm: LLMClient,
        config: AppConfig,
    ) -> PortfolioDecision:
        payload = llm.generate_json(
            task_name="portfolio_manager",
            model=config.model_for("portfolio"),
            system_prompt=(
                "You are the portfolio manager. Grant final approval only if the trade and risk profile fit the overall book."
            ),
            user_prompt=(
                "Return JSON with keys: approved, final_action, final_position_pct, rationale, conditions.\n\n"
                f"Analyst consensus:\n{_compact_json([report.to_dict() for report in reports])}\n\n"
                f"Trade decision:\n{_compact_json(trade.to_dict())}\n\n"
                f"Risk review:\n{_compact_json(risk.to_dict())}\n\n"
                f"Macro snapshot:\n{_compact_json(snapshot.macro_snapshot)}"
            ),
        )
        final_action = str(payload.get("final_action") or trade.action).upper()
        if final_action not in {"BUY", "SELL", "HOLD"}:
            final_action = "HOLD"
        return PortfolioDecision(
            approved=bool(payload.get("approved", False)),
            final_action=final_action,
            final_position_pct=max(0.0, _float(payload.get("final_position_pct"), 0.0)),
            rationale=str(payload.get("rationale") or "No rationale provided."),
            conditions=_list(payload.get("conditions")),
            raw=payload,
        )
