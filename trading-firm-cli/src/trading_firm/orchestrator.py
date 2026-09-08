from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from trading_firm.agents import (
    AnalystAgent,
    PortfolioManagerAgent,
    ResearchAgent,
    RiskManagerAgent,
    TraderAgent,
)
from trading_firm.config import AppConfig
from trading_firm.data import MarketDataCollector
from trading_firm.llm import build_llm_client
from trading_firm.models import WorkflowResult


class TradingFirmOrchestrator:
    def __init__(self, config: AppConfig, force_mock: bool = False) -> None:
        self.config = config
        self.collector = MarketDataCollector(config)
        self.llm = build_llm_client(config, force_mock=force_mock)

    def run(self, symbol: str, company_name: str | None = None) -> WorkflowResult:
        snapshot = self.collector.build_snapshot(
            symbol=symbol, company_name=company_name
        )

        analysts = [
            AnalystAgent(
                "fundamentals",
                "Assess financial quality, valuation, and balance-sheet strength.",
            ),
            AnalystAgent(
                "sentiment",
                "Assess crowd mood, retail positioning, and narrative momentum from public social chatter.",
            ),
            AnalystAgent(
                "technical",
                "Assess trend structure, momentum, volatility, and tactical timing.",
            ),
            AnalystAgent(
                "news",
                "Assess company news and macro events that can affect the trade over the next few weeks.",
            ),
        ]

        with ThreadPoolExecutor(max_workers=len(analysts)) as executor:
            analyst_reports = list(
                executor.map(
                    lambda agent: agent.run(snapshot, self.llm, self.config), analysts
                )
            )

        researchers = [
            ResearchAgent("bullish", "bullish"),
            ResearchAgent("bearish", "bearish"),
        ]
        with ThreadPoolExecutor(max_workers=len(researchers)) as executor:
            debate_memos = list(
                executor.map(
                    lambda agent: agent.run(
                        snapshot, analyst_reports, self.llm, self.config
                    ),
                    researchers,
                )
            )

        trader = TraderAgent()
        risk_manager = RiskManagerAgent()
        portfolio_manager = PortfolioManagerAgent()

        trade_decision = trader.run(
            snapshot, analyst_reports, debate_memos, self.llm, self.config
        )
        risk_review = risk_manager.run(snapshot, trade_decision, self.llm, self.config)
        portfolio_decision = portfolio_manager.run(
            snapshot,
            analyst_reports,
            trade_decision,
            risk_review,
            self.llm,
            self.config,
        )

        return WorkflowResult(
            snapshot=snapshot,
            analyst_reports=analyst_reports,
            debate_memos=debate_memos,
            trade_decision=trade_decision,
            risk_review=risk_review,
            portfolio_decision=portfolio_decision,
        )
