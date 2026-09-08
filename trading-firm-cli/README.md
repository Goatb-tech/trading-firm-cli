# Trading Firm CLI

`trading-firm` is a command-line Python framework that models a small trading desk made up of specialized AI agents.

The workflow is:

1. Fundamentals analyst reviews company financial quality and valuation.
2. Sentiment analyst reviews public social-media discussion.
3. Technical analyst reviews price structure and momentum.
4. News analyst reviews company and macro headlines.
5. Bullish and bearish researchers debate the evidence.
6. A trader proposes a trade.
7. A risk manager applies constraints.
8. A portfolio manager gives final approval.

## Features

- Command-line workflow for a complete analyst-to-approval chain
- Pluggable LLM backend through LiteLLM for OpenAI, Gemini, Claude, OpenRouter, and others
- `.env` driven configuration for provider keys and per-agent model overrides
- Market data collection using Yahoo Finance, Reuters RSS feeds, and public Reddit search results
- Mock mode for local dry runs without calling any LLM provider
- Docker and docker-compose support

## Quick Start

1. Create and activate a Python 3.11+ environment.
2. Install the project.

```bash
pip install -e .
```

3. Copy the sample environment file.

```bash
cp .env.example .env
```

4. Edit `.env`.

- Set `LLM_BACKEND=litellm` to use a real provider.
- Add the provider API key you want to use.
- Set `LLM_MODEL` to any LiteLLM-compatible model string.

Examples:

- OpenAI: `LLM_MODEL=gpt-4.1-mini`
- Gemini: `LLM_MODEL=gemini/gemini-2.0-flash`
- Claude: `LLM_MODEL=claude-3-5-sonnet-20241022`
- OpenRouter: `LLM_MODEL=openrouter/openai/gpt-4.1-mini`

5. Run the workflow.

```bash
trading-firm AAPL --company "Apple Inc."
```

## CLI Usage

```bash
trading-firm SYMBOL [--company NAME] [--model MODEL] [--json-output report.json] [--markdown-output report.md] [--mock]
```

Examples:

```bash
trading-firm NVDA --json-output nvda.json --markdown-output nvda.md
trading-firm MSFT --model gemini/gemini-2.0-flash
trading-firm TSLA --mock
```

## Environment Variables

Core settings:

- `LLM_BACKEND`: `mock` or `litellm`
- `LLM_MODEL`: default model for all agents
- `FALLBACK_MODELS`: comma-separated fallback list
- `REQUEST_TIMEOUT`: request timeout in seconds
- `TEMPERATURE`: model temperature
- `LOOKBACK_PERIOD`: Yahoo Finance history window, such as `3mo` or `6mo`
- `MAX_NEWS_ITEMS`: max company and macro news items
- `MAX_SOCIAL_POSTS`: max Reddit posts to score

Provider keys:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`

Per-agent overrides:

- `FUNDAMENTALS_MODEL`
- `SENTIMENT_MODEL`
- `TECHNICAL_MODEL`
- `NEWS_MODEL`
- `BULLISH_MODEL`
- `BEARISH_MODEL`
- `TRADER_MODEL`
- `RISK_MODEL`
- `PORTFOLIO_MODEL`

## Docker

Build and run:

```bash
docker build -t trading-firm .
docker run --rm --env-file .env trading-firm trading-firm AAPL
```

Or with compose:

```bash
docker compose run --rm trading-firm trading-firm AAPL
```

## Notes

- This project is a research and workflow framework, not a live execution engine.
- Yahoo Finance and public Reddit endpoints can rate-limit or change format over time.
- Mock mode is useful for smoke tests, but it does not produce real investment reasoning.
