from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings
from app.infrastructure.config.graph_loader import YamlGraphRegistry, load_yaml_graphs
from app.infrastructure.integrations.openhands import OpenHandsAdapter
from app.infrastructure.persistence.mongo import MongoClientProvider, MongoGraphRunRepository
from app.infrastructure.tools.mcp_client import McpToolsProvider


@dataclass
class ApplicationContainer:
    settings: Settings
    llm: BaseChatModel
    mcp_tools_provider: McpToolsProvider
    yaml_graph_registry: YamlGraphRegistry
    mongo_provider: MongoClientProvider
    run_repository: MongoGraphRunRepository
    openhands: OpenHandsAdapter

    async def startup(self) -> None:
        await self.mcp_tools_provider.start()

    async def shutdown(self) -> None:
        await self.mcp_tools_provider.stop()
        await self.mongo_provider.close()


_ANTHROPIC_DEFAULT_MODEL = "claude-opus-4-6"
_OPENAI_DEFAULT_MODEL = "gpt-4o"
_GOOGLE_DEFAULT_MODEL = "gemini-2.0-flash"


def build_llm(settings: Settings) -> BaseChatModel:
    provider = (settings.llm_provider or "").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.llm_model or _ANTHROPIC_DEFAULT_MODEL,
            api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model or _OPENAI_DEFAULT_MODEL,
            api_key=settings.openai_api_key,  # type: ignore[arg-type]
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.llm_model or _GOOGLE_DEFAULT_MODEL,
            google_api_key=settings.google_api_key,  # type: ignore[arg-type]
        )

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    return FakeMessagesListChatModel(
        responses=[AIMessage(content="LLM not configured. Set LLM_PROVIDER and the matching API key.")]
    )


def build_container(settings: Settings) -> ApplicationContainer:
    llm = build_llm(settings)
    mcp_tools_provider = McpToolsProvider(settings)
    openhands = OpenHandsAdapter(settings)
    mongo_provider = MongoClientProvider(settings)
    run_repository = mongo_provider.get_repository()
    yaml_graph_registry = load_yaml_graphs(
        settings.graph_definitions_path,
        llm=llm,
        mcp_tools_provider=mcp_tools_provider,
        openhands=openhands,
        run_repository=run_repository,
    )
    return ApplicationContainer(
        settings=settings,
        llm=llm,
        mcp_tools_provider=mcp_tools_provider,
        yaml_graph_registry=yaml_graph_registry,
        mongo_provider=mongo_provider,
        run_repository=run_repository,
        openhands=openhands,
    )
