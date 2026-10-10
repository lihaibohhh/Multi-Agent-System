"""Default Agent runtime composition for production graph nodes.

The workflow compatibility facade may provide alternate dependencies in tests,
but new code should import the shared runner and context helpers from here.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ..core.config import settings
from ..retrieval import retrieve_evidence
from ..sections.model_output import invoke_checked
from .context import AgentContext, create_resumable_agent_context
from .events import dispatch_agent_event
from .runtime import AgentConfigurationError, AgentRunner
from .spec import AgentSpecAny


async def call_model(system: str, prompt: str, schema=None, *, validator=None, context=None):
    """Invoke the configured section model through the checked output gateway."""
    from ..utils.llm import load_chat_model

    model = load_chat_model(settings.agent.section_model)
    return await invoke_checked(
        model,
        system,
        prompt,
        schema,
        validator=validator,
        context=context,
    )


def resolve_agent_model(spec: AgentSpecAny):
    """Resolve an Agent's symbolic model reference."""
    if spec.model_ref != "section_model":
        raise AgentConfigurationError(
            f"Agent '{spec.name}' 使用了未知模型引用：{spec.model_ref}"
        )
    return call_model


async def retrieve_tool(request):
    """Late-bound read-only retrieval adapter used by AgentRunner."""
    return await retrieve_evidence(request)


def create_agent_runner(
    model_resolver=resolve_agent_model,
    *,
    retrieval=retrieve_tool,
) -> AgentRunner:
    """Compose an Agent runner without coupling graph nodes to providers."""
    return AgentRunner(
        model_resolver,
        tools={"retrieve_evidence": retrieval},
        event_sink=dispatch_agent_event,
    )


agent_runner = create_agent_runner()


def agent_context(
    config: RunnableConfig | None = None,
    *,
    section_id: str | None = None,
) -> AgentContext:
    """Build a run-local Agent identity from LangGraph configuration."""
    run_id = str((config or {}).get("configurable", {}).get("thread_id", "")).strip()
    return AgentContext.create(run_id or "standalone", section_id=section_id)


async def resumable_agent_context(
    agent_name: str,
    config: RunnableConfig | None = None,
    *,
    section_id: str | None = None,
) -> AgentContext:
    """Build an Agent context linked to the last resumable execution."""
    run_id = str((config or {}).get("configurable", {}).get("thread_id", "")).strip()
    return await create_resumable_agent_context(
        run_id or "standalone",
        agent_name,
        section_id=section_id,
    )


__all__ = [
    "agent_context",
    "agent_runner",
    "call_model",
    "create_agent_runner",
    "resolve_agent_model",
    "resumable_agent_context",
    "retrieve_tool",
]
