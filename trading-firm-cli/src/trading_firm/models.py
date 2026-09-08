from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ContextSnapshot:
    symbol: str
    company_name: str
    fundamentals: dict[str, Any]
    price_summary: dict[str, Any]
    technicals: dict[str, Any]
    social_sentiment: dict[str, Any]
    company_news: list[dict[str, Any]]
    macro_news: list[dict[str, Any]]
    macro_snapshot: dict[str, Any]
    data_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnalystReport:
    agent: str
    title: str
    stance: str
    score: int
    confidence: int
    summary: str
    key_points: list[str]
    risks: list[str]
    data_points: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DebateMemo:
    agent: str
    stance: str
    conviction: int
    thesis: str
    supporting_points: list[str]
    rebuttals: list[str]
    watch_items: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TradeDecision:
    action: str
    confidence: int
    position_pct: float
    time_horizon: str
    thesis: str
    entry_rationale: list[str]
    exit_conditions: list[str]
    primary_risks: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RiskReview:
    approved: bool
    max_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    risk_notes: list[str]
    required_changes: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PortfolioDecision:
    approved: bool
    final_action: str
    final_position_pct: float
    rationale: str
    conditions: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowResult:
    snapshot: ContextSnapshot
    analyst_reports: list[AnalystReport]
    debate_memos: list[DebateMemo]
    trade_decision: TradeDecision
    risk_review: RiskReview
    portfolio_decision: PortfolioDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "analyst_reports": [report.to_dict() for report in self.analyst_reports],
            "debate_memos": [memo.to_dict() for memo in self.debate_memos],
            "trade_decision": self.trade_decision.to_dict(),
            "risk_review": self.risk_review.to_dict(),
            "portfolio_decision": self.portfolio_decision.to_dict(),
        }
