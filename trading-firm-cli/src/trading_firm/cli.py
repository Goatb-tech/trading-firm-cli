from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading-firm",
        description="Run a multi-agent trading workflow from the command line.",
    )
    parser.add_argument(
        "symbol", nargs="?", help="Ticker symbol, for example AAPL or MSFT"
    )
    parser.add_argument("--company", help="Optional company name override")
    parser.add_argument(
        "--json-output", help="Write the full workflow output to a JSON file"
    )
    parser.add_argument("--markdown-output", help="Write a Markdown summary to a file")
    parser.add_argument("--model", help="Override the default model for all agents")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the mock LLM backend even if real provider keys are configured",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.symbol:
        parser.print_help()
        return 0

    from trading_firm.config import AppConfig
    from trading_firm.orchestrator import TradingFirmOrchestrator

    config = AppConfig.from_env()
    if args.model:
        config.default_model = args.model

    orchestrator = TradingFirmOrchestrator(config, force_mock=args.mock)
    result = orchestrator.run(symbol=args.symbol, company_name=args.company)

    markdown_report = render_markdown(result.to_dict())
    print(markdown_report)

    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown_report, encoding="utf-8")
    return 0


def render_markdown(result: dict) -> str:
    snapshot = result["snapshot"]
    trade = result["trade_decision"]
    risk = result["risk_review"]
    portfolio = result["portfolio_decision"]

    lines = [
        f"# {snapshot['symbol']} Trading Committee",
        "",
        f"**Company:** {snapshot['company_name']}",
        f"**Trader Recommendation:** {trade['action']} {trade['position_pct']:.2f}%",
        f"**Risk Approved:** {'yes' if risk['approved'] else 'no'}",
        f"**Portfolio Approved:** {'yes' if portfolio['approved'] else 'no'}",
        "",
        "## Analyst Team",
    ]

    for report in result["analyst_reports"]:
        lines.extend(
            [
                f"### {report['agent'].title()} Analyst",
                f"- Stance: {report['stance']} ({report['score']})",
                f"- Confidence: {report['confidence']}",
                f"- Summary: {report['summary']}",
            ]
        )
        for point in report["key_points"]:
            lines.append(f"- {point}")
        if report["risks"]:
            lines.append("- Risks: " + "; ".join(report["risks"]))
        lines.append("")

    lines.append("## Debate Desk")
    for memo in result["debate_memos"]:
        lines.extend(
            [
                f"### {memo['agent'].title()} Researcher",
                f"- Conviction: {memo['conviction']}",
                f"- Thesis: {memo['thesis']}",
            ]
        )
        for point in memo["supporting_points"]:
            lines.append(f"- {point}")
        lines.append("")

    lines.extend(
        [
            "## Trade Plan",
            f"- Action: {trade['action']}",
            f"- Position Size: {trade['position_pct']:.2f}%",
            f"- Horizon: {trade['time_horizon']}",
            f"- Thesis: {trade['thesis']}",
            f"- Entry: {'; '.join(trade['entry_rationale'])}",
            f"- Exit: {'; '.join(trade['exit_conditions'])}",
            "",
            "## Risk Review",
            f"- Approved: {'yes' if risk['approved'] else 'no'}",
            f"- Max Position: {risk['max_position_pct']:.2f}%",
            f"- Stop Loss: {risk['stop_loss_pct']:.2f}%",
            f"- Take Profit: {risk['take_profit_pct']:.2f}%",
            f"- Notes: {'; '.join(risk['risk_notes'])}",
            "",
            "## Portfolio Decision",
            f"- Approved: {'yes' if portfolio['approved'] else 'no'}",
            f"- Final Action: {portfolio['final_action']}",
            f"- Final Position: {portfolio['final_position_pct']:.2f}%",
            f"- Rationale: {portfolio['rationale']}",
        ]
    )

    if snapshot.get("data_warnings"):
        lines.extend(["", "## Data Warnings"])
        for warning in snapshot["data_warnings"]:
            lines.append(f"- {warning}")

    return "\n".join(lines)
