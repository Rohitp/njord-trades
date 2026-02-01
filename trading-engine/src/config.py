from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "trading"
    user: str = "postgres"
    password: str = ""

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALPACA_")

    api_key: str = ""
    secret_key: str = ""
    base_url: str = "https://paper-api.alpaca.markets"  # Paper trading default
    paper: bool = True


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    default_provider: str = "anthropic"
    data_agent_model: str = "claude-sonnet-4-20250514"
    risk_agent_model: str = "claude-sonnet-4-20250514"
    validator_model: str = "claude-sonnet-4-20250514"
    meta_agent_model: str = "claude-sonnet-4-20250514"


class TradingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADING_")

    max_position_pct: float = Field(default=0.20, description="Max 20% per position")
    max_sector_pct: float = Field(default=0.30, description="Max 30% per sector")
    max_positions: int = Field(default=10, description="Max concurrent positions")
    drawdown_halt_pct: float = Field(default=0.20, description="Halt at 20% drawdown")
    consecutive_loss_halt: int = Field(default=10, description="Halt after 10 losses")
    sharpe_halt_days: int = Field(default=30, description="Days of negative Sharpe to halt")


class AlertSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ALERT_")

    discord_webhook_url: str = ""
    email_enabled: bool = False
    email_from: str = ""
    email_to: str = ""
    aws_region: str = "us-east-1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    environment: str = Field(default="development", description="development|staging|production")
    debug: bool = False

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    alpaca: AlpacaSettings = Field(default_factory=AlpacaSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)


settings = Settings()
