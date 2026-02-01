from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5556
    name: str = "trading"
    user: str = "postgres"
    password: str = ""
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout seconds")
    echo: bool = Field(default=False, description="Echo SQL queries")

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
    # Fixed model names - use actual Claude model identifiers
    data_agent_model: str = "claude-3-5-sonnet-20241022"
    risk_agent_model: str = "claude-3-5-sonnet-20241022"
    validator_model: str = "claude-3-5-sonnet-20241022"
    meta_agent_model: str = "claude-3-opus-20240229"  # Highest quality for Meta-Agent
    
    # Retry settings (required by instructions)
    max_retries: int = Field(default=3, description="Max retry attempts for LLM calls")
    retry_backoff_base: float = Field(default=2.0, description="Exponential backoff base (seconds)")
    retry_backoff_max: float = Field(default=8.0, description="Max backoff delay (seconds)")


class TradingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRADING_")

    # Hard constraints (cannot be overridden by LLM)
    max_position_pct: float = Field(default=0.20, description="Max 20% per position")
    max_sector_pct: float = Field(default=0.30, description="Max 30% per sector")
    max_positions: int = Field(default=10, description="Max concurrent positions")
    
    # Circuit breaker triggers
    drawdown_halt_pct: float = Field(default=0.20, description="Halt at 20% drawdown")
    consecutive_loss_halt: int = Field(default=10, description="Halt after 10 losses")
    sharpe_halt_days: int = Field(default=30, description="Days of negative Sharpe to halt")
    
    # Circuit breaker auto-resume conditions (required by instructions)
    drawdown_resume_pct: float = Field(default=0.15, description="Resume when drawdown <15%")
    win_streak_resume: int = Field(default=3, description="Resume after 3 wins")
    sharpe_resume_threshold: float = Field(default=0.3, description="Resume when Sharpe >0.3")
    sharpe_resume_days: int = Field(default=7, description="Sharpe threshold must hold for 7 days")
    
    # Initial capital (required by instructions)
    initial_capital: float = Field(default=500.0, description="Initial capital in GBP")
    currency: str = Field(default="GBP", description="Base currency")
    
    # Watchlist symbols (required by instructions)
    watchlist_symbols: list[str] = Field(
        default_factory=lambda: ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN"],
        description="Symbols to scan for signals"
    )


class VectorDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTOR_DB_")
    
    provider: str = Field(default="chroma", description="chroma|qdrant")
    chroma_path: str = Field(default="./chroma_db", description="Chroma persistence path")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_api_key: str = ""
    collection_name: str = Field(default="trading_events", description="Vector collection name")


class LangSmithSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LANGSMITH_")
    
    api_key: str = ""
    project: str = Field(default="trading-system", description="LangSmith project name")
    tracing_enabled: bool = Field(default=True, description="Enable LangSmith tracing")


class SchedulingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SCHEDULE_")
    
    scan_times: list[str] = Field(
        default=["11:00", "14:30"],
        description="Scheduled scan times (EST, HH:MM format)"
    )
    timezone: str = Field(default="America/New_York", description="Trading timezone")
    market_hours_only: bool = Field(default=True, description="Only run during market hours")
    weekdays_only: bool = Field(default=True, description="Only run on weekdays")


class EventMonitorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVENT_MONITOR_")
    
    price_move_threshold_pct: float = Field(default=0.05, description="5% price move trigger")
    move_window_minutes: int = Field(default=15, description="Time window for price move")
    cooldown_minutes: int = Field(default=15, description="Cooldown between event scans")
    poll_interval_seconds: int = Field(default=60, description="Polling interval")
    stocks_only: bool = Field(default=True, description="Stocks only (no options in event scans)")


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOG_")
    
    level: str = Field(default="INFO", description="Log level: DEBUG|INFO|WARNING|ERROR")
    format: str = Field(default="json", description="json|console")
    file_path: str = Field(default="logs/trading.log", description="Log file path")
    structured: bool = Field(default=True, description="Use structured logging (structlog)")


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
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    scheduling: SchedulingSettings = Field(default_factory=SchedulingSettings)
    event_monitor: EventMonitorSettings = Field(default_factory=EventMonitorSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


settings = Settings()
