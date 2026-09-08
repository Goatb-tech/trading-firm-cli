from __future__ import annotations

from dataclasses import dataclass, field
import os

from dotenv import load_dotenv


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class AppConfig:
    llm_backend: str = "mock"
    default_model: str = "gpt-4.1-mini"
    fallback_models: list[str] = field(default_factory=list)
    request_timeout: int = 60
    temperature: float = 0.2
    max_news_items: int = 6
    max_social_posts: int = 10
    lookback_period: str = "6mo"
    agent_models: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()

        agent_models = {
            "fundamentals": os.getenv("FUNDAMENTALS_MODEL", "").strip(),
            "sentiment": os.getenv("SENTIMENT_MODEL", "").strip(),
            "technical": os.getenv("TECHNICAL_MODEL", "").strip(),
            "news": os.getenv("NEWS_MODEL", "").strip(),
            "bullish": os.getenv("BULLISH_MODEL", "").strip(),
            "bearish": os.getenv("BEARISH_MODEL", "").strip(),
            "trader": os.getenv("TRADER_MODEL", "").strip(),
            "risk": os.getenv("RISK_MODEL", "").strip(),
            "portfolio": os.getenv("PORTFOLIO_MODEL", "").strip(),
        }

        return cls(
            llm_backend=os.getenv("LLM_BACKEND", "mock").strip().lower(),
            default_model=os.getenv("LLM_MODEL", "gpt-4.1-mini").strip(),
            fallback_models=_split_csv(os.getenv("FALLBACK_MODELS")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
            temperature=float(os.getenv("TEMPERATURE", "0.2")),
            max_news_items=int(os.getenv("MAX_NEWS_ITEMS", "6")),
            max_social_posts=int(os.getenv("MAX_SOCIAL_POSTS", "10")),
            lookback_period=os.getenv("LOOKBACK_PERIOD", "6mo").strip(),
            agent_models={key: value for key, value in agent_models.items() if value},
        )

    def model_for(self, role: str, explicit_model: str | None = None) -> str:
        if explicit_model:
            return explicit_model
        return self.agent_models.get(role, self.default_model)
