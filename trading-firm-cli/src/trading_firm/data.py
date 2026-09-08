from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import Any

import feedparser
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

from trading_firm.config import AppConfig
from trading_firm.models import ContextSnapshot


class MarketDataCollector:
    REDDIT_URL = "https://www.reddit.com/search.json"
    MACRO_RSS_FEEDS = [
        "https://news.google.com/rss/search?q=Federal+Reserve+OR+inflation+OR+interest+rates+when:7d&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=jobs+report+OR+GDP+OR+oil+prices+OR+treasury+yields+when:7d&hl=en-US&gl=US&ceid=US:en",
    ]

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradingFirmCLI/0.1"})
        self.sentiment = SentimentIntensityAnalyzer()

    def build_snapshot(
        self, symbol: str, company_name: str | None = None
    ) -> ContextSnapshot:
        warnings: list[str] = []
        ticker = yf.Ticker(symbol)
        info = self._safe_info(ticker)
        resolved_name = (
            company_name or info.get("shortName") or info.get("longName") or symbol
        )

        fundamentals = self._collect_fundamentals(ticker, info)
        technicals, price_summary = self._collect_technicals(symbol)
        company_news = self._collect_company_news(ticker)
        macro_news = self._collect_macro_news()
        social_sentiment = self._collect_social_sentiment(symbol, resolved_name)
        macro_snapshot = self._collect_macro_snapshot()

        if not fundamentals:
            warnings.append("Yahoo Finance did not return a fundamentals snapshot.")
        if not company_news:
            warnings.append(
                "No company-specific news items were available from Yahoo Finance."
            )
        if not social_sentiment.get("posts_analyzed"):
            warnings.append(
                "No public Reddit posts were collected for sentiment analysis."
            )
        if not macro_news:
            warnings.append(
                "No macro headlines were collected from the configured RSS feeds."
            )

        return ContextSnapshot(
            symbol=symbol.upper(),
            company_name=resolved_name,
            fundamentals=fundamentals,
            price_summary=price_summary,
            technicals=technicals,
            social_sentiment=social_sentiment,
            company_news=company_news,
            macro_news=macro_news,
            macro_snapshot=macro_snapshot,
            data_warnings=warnings,
        )

    def _safe_info(self, ticker: yf.Ticker) -> dict[str, Any]:
        try:
            return ticker.info or {}
        except Exception:
            return {}

    def _collect_fundamentals(
        self, ticker: yf.Ticker, info: dict[str, Any]
    ) -> dict[str, Any]:
        snapshot = {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": self._clean_number(info.get("marketCap")),
            "enterprise_value": self._clean_number(info.get("enterpriseValue")),
            "trailing_pe": self._clean_number(info.get("trailingPE")),
            "forward_pe": self._clean_number(info.get("forwardPE")),
            "price_to_book": self._clean_number(info.get("priceToBook")),
            "profit_margin": self._clean_number(info.get("profitMargins")),
            "gross_margin": self._clean_number(info.get("grossMargins")),
            "operating_margin": self._clean_number(info.get("operatingMargins")),
            "revenue_growth": self._clean_number(info.get("revenueGrowth")),
            "earnings_growth": self._clean_number(info.get("earningsGrowth")),
            "return_on_equity": self._clean_number(info.get("returnOnEquity")),
            "debt_to_equity": self._clean_number(info.get("debtToEquity")),
            "free_cashflow": self._clean_number(info.get("freeCashflow")),
            "target_mean_price": self._clean_number(info.get("targetMeanPrice")),
            "recommendation_key": info.get("recommendationKey"),
        }

        quarterly_income = self._statement_snapshot(
            ticker.quarterly_income_stmt, ["Total Revenue", "Net Income"]
        )
        quarterly_balance = self._statement_snapshot(
            ticker.quarterly_balance_sheet, ["Total Debt", "Stockholders Equity"]
        )
        quarterly_cashflow = self._statement_snapshot(
            ticker.quarterly_cashflow, ["Operating Cash Flow", "Free Cash Flow"]
        )

        if quarterly_income:
            snapshot["quarterly_income_trend"] = quarterly_income
        if quarterly_balance:
            snapshot["quarterly_balance_trend"] = quarterly_balance
        if quarterly_cashflow:
            snapshot["quarterly_cashflow_trend"] = quarterly_cashflow

        return {
            key: value
            for key, value in snapshot.items()
            if value is not None and value != {}
        }

    def _statement_snapshot(
        self, statement: Any, rows: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        if statement is None or getattr(statement, "empty", True):
            return {}

        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row not in statement.index:
                continue
            series = statement.loc[row].dropna().head(4)
            result[row] = [
                {
                    "period": str(column.date() if hasattr(column, "date") else column),
                    "value": self._clean_number(value),
                }
                for column, value in series.items()
            ]
        return result

    def _collect_technicals(self, symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            history = yf.download(
                symbol,
                period=self.config.lookback_period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception:
            return {}, {}

        if history.empty:
            return {}, {}

        close = self._series(history, "Close")
        volume = self._series(history, "Volume")
        if close.empty:
            return {}, {}

        last_price = float(close.iloc[-1])
        sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        sma200 = (
            float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
        )
        rsi14 = self._rsi(close, 14)
        macd_line, macd_signal = self._macd(close)
        recent_return = (
            ((close.iloc[-1] / close.iloc[max(0, len(close) - 22)] - 1) * 100)
            if len(close) > 21
            else None
        )
        period_return = (
            ((close.iloc[-1] / close.iloc[0] - 1) * 100) if len(close) > 1 else None
        )
        volatility = (
            close.pct_change().dropna().tail(21).std() * (252**0.5) * 100
            if len(close) > 21
            else None
        )
        avg_volume = float(volume.tail(20).mean()) if not volume.empty else None
        last_volume = float(volume.iloc[-1]) if not volume.empty else None
        volume_ratio = (last_volume / avg_volume) if avg_volume else None

        technicals = {
            "last_price": round(last_price, 2),
            "sma20": self._round_or_none(sma20),
            "sma50": self._round_or_none(sma50),
            "sma200": self._round_or_none(sma200),
            "rsi14": self._round_or_none(rsi14),
            "macd_line": self._round_or_none(macd_line),
            "macd_signal": self._round_or_none(macd_signal),
            "trend": self._trend_label(last_price, sma20, sma50),
            "distance_from_52w_high_pct": self._distance_pct(
                last_price,
                float(close.tail(252).max() if len(close) >= 2 else close.max()),
            ),
            "distance_from_52w_low_pct": self._distance_pct(
                last_price,
                float(close.tail(252).min() if len(close) >= 2 else close.min()),
            ),
            "volume_ratio_vs_20d_avg": self._round_or_none(volume_ratio),
        }

        price_summary = {
            "current_price": round(last_price, 2),
            "one_month_return_pct": self._round_or_none(recent_return),
            "period_return_pct": self._round_or_none(period_return),
            "annualized_volatility_pct": self._round_or_none(volatility),
            "average_20d_volume": self._round_or_none(avg_volume),
            "latest_volume": self._round_or_none(last_volume),
            "last_close_date": str(close.index[-1].date()),
        }

        return technicals, price_summary

    def _collect_company_news(self, ticker: yf.Ticker) -> list[dict[str, Any]]:
        try:
            raw_items = ticker.news or []
        except Exception:
            return []

        items: list[dict[str, Any]] = []
        for item in raw_items[: self.config.max_news_items]:
            content = item.get("content") if isinstance(item, dict) else None
            title = item.get("title") or (content or {}).get("title")
            summary = item.get("summary") or (content or {}).get("summary")
            publisher = (
                item.get("publisher")
                or (content or {}).get("provider")
                or (content or {}).get("pubName")
            )
            link = item.get("link") or (content or {}).get("canonicalUrl", {}).get(
                "url"
            )
            published = item.get("providerPublishTime") or (content or {}).get(
                "pubDate"
            )
            items.append(
                {
                    "title": title,
                    "summary": summary,
                    "publisher": publisher,
                    "link": link,
                    "published": self._normalize_timestamp(published),
                }
            )
        return [item for item in items if item.get("title")]

    def _collect_macro_news(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for url in self.MACRO_RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
            except Exception:
                continue
            for entry in feed.entries[: self.config.max_news_items]:
                items.append(
                    {
                        "title": getattr(entry, "title", None),
                        "summary": getattr(entry, "summary", None),
                        "publisher": (getattr(entry, "source", {}) or {}).get("title")
                        if hasattr(entry, "source")
                        else None,
                        "link": getattr(entry, "link", None),
                        "published": getattr(entry, "published", None),
                    }
                )
                if len(items) >= self.config.max_news_items:
                    return items
        return items

    def _collect_social_sentiment(
        self, symbol: str, company_name: str
    ) -> dict[str, Any]:
        query = f'("{symbol}") OR ("{company_name}")'
        try:
            response = self.session.get(
                self.REDDIT_URL,
                params={
                    "q": query,
                    "sort": "new",
                    "limit": self.config.max_social_posts,
                },
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return {
                "mood": "unknown",
                "posts_analyzed": 0,
                "average_compound_score": 0.0,
                "top_snippets": [],
            }

        posts = payload.get("data", {}).get("children", [])
        scored_posts: list[dict[str, Any]] = []
        scores: list[float] = []

        for post in posts[: self.config.max_social_posts]:
            data = post.get("data", {})
            text = " ".join(filter(None, [data.get("title"), data.get("selftext")]))
            if not text.strip():
                continue
            compound = self.sentiment.polarity_scores(text)["compound"]
            scores.append(compound)
            scored_posts.append(
                {
                    "title": data.get("title"),
                    "subreddit": data.get("subreddit"),
                    "score": data.get("score"),
                    "sentiment": round(compound, 3),
                    "url": f"https://reddit.com{data.get('permalink', '')}",
                }
            )

        average_score = mean(scores) if scores else 0.0
        mood = (
            "bullish"
            if average_score > 0.1
            else "bearish"
            if average_score < -0.1
            else "neutral"
        )

        return {
            "mood": mood,
            "posts_analyzed": len(scores),
            "average_compound_score": round(average_score, 3),
            "top_snippets": scored_posts[: min(5, len(scored_posts))],
        }

    def _collect_macro_snapshot(self) -> dict[str, Any]:
        symbols = {
            "S&P 500": "^GSPC",
            "Nasdaq": "^IXIC",
            "VIX": "^VIX",
            "US 10Y Yield": "^TNX",
            "Crude Oil": "CL=F",
            "Gold": "GC=F",
            "US Dollar Index": "DX-Y.NYB",
        }
        snapshot: dict[str, Any] = {}

        for label, symbol in symbols.items():
            try:
                history = yf.download(
                    symbol,
                    period="1mo",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
            except Exception:
                continue
            if history.empty:
                continue
            close = self._series(history, "Close")
            if close.empty:
                continue
            last = float(close.iloc[-1])
            one_month_change = (
                ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) > 1 else 0.0
            )
            snapshot[label] = {
                "last": round(last, 2),
                "one_month_change_pct": round(one_month_change, 2),
            }
        return snapshot

    def _series(self, history: pd.DataFrame, column: str) -> pd.Series:
        series = history[column]
        if isinstance(series, pd.DataFrame):
            return series.iloc[:, 0].dropna()
        return series.dropna()

    def _clean_number(self, value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 4)
        return value

    def _normalize_timestamp(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        return str(value)

    def _round_or_none(self, value: float | None) -> float | None:
        return None if value is None else round(float(value), 2)

    def _distance_pct(self, current: float, anchor: float) -> float | None:
        if not anchor:
            return None
        return round(((current / anchor) - 1) * 100, 2)

    def _trend_label(
        self, last_price: float, sma20: float | None, sma50: float | None
    ) -> str:
        if sma20 is None or sma50 is None:
            return "unknown"
        if last_price > sma20 > sma50:
            return "bullish"
        if last_price < sma20 < sma50:
            return "bearish"
        return "mixed"

    def _rsi(self, close: pd.Series, period: int) -> float | None:
        if len(close) <= period:
            return None
        delta = close.diff().dropna()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(period).mean().iloc[-1]
        avg_loss = losses.rolling(period).mean().iloc[-1]
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _macd(self, close: pd.Series) -> tuple[float | None, float | None]:
        if len(close) < 26:
            return None, None
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal = macd_line.ewm(span=9, adjust=False).mean()
        return float(macd_line.iloc[-1]), float(signal.iloc[-1])
