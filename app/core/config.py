from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Development Orchestration System"
    environment: Literal["local", "dev", "prod", "test"] = "local"
    api_prefix: str = "/api/v1"
    debug: bool = False
    workflow_definitions_path: str = Field(default="workflows", alias="WORKFLOW_DEFINITIONS_PATH")
    mongodb_uri: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URI")
    mongodb_database: str = Field(default="langgraph_backend", alias="MONGODB_DATABASE")
    workflow_runs_collection: str = "workflow_runs"
    openhands_base_url: str = Field(default="http://openhands:3000", alias="OPENHANDS_BASE_URL")
    openhands_api_key: str | None = Field(default=None, alias="OPENHANDS_API_KEY")
    openhands_timeout_seconds: float = Field(default=60.0, alias="OPENHANDS_TIMEOUT_SECONDS")
    openhands_mock_mode: bool = Field(default=True, alias="OPENHANDS_MOCK_MODE")
    langserve_path: str = Field(default="/langserve/workflow-runner", alias="LANGSERVE_PATH")
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
