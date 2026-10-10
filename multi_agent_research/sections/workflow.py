"""Compatibility facade for chapter workflow nodes.

Implementations live in focused ``nodes``, ``context_builder``, ``transitions``,
``policies``, and ``report_assembler`` modules. This facade preserves the
historical import and monkeypatch surface while callers migrate.
"""

from langchain_core.runnables import RunnableConfig

from ..agents import bootstrap as agent_bootstrap
from ..agents.events import dispatch_agent_event
from ..agents.runtime import AgentConfigurationError, AgentRunner
from ..core.config import settings
from ..retrieval import retrieve_evidence
from . import context_builder, transitions
from .model_output import invoke_checked
from .nodes import planning as planning_nodes
from .nodes import editorial as editorial_nodes
from .nodes import report as report_nodes
from .nodes import research as research_nodes
from .nodes import review as review_nodes
from .nodes import writing as writing_nodes
from . import policies
from .report_assembler import assemble_sections


# Historical helper aliases remain available during the import migration.
_agent_context = agent_bootstrap.agent_context
_resumable_agent_context = agent_bootstrap.resumable_agent_context
_current = context_builder.current_section
_dependencies = context_builder.dependency_revisions
_evidence_text = context_builder.evidence_text
_parent_view = context_builder.parent_view
_prior_context = context_builder.prior_context
_prompt = context_builder.section_prompt
_policy = policies.section_policy
_select_sources = policies.select_sources
_archive_draft = transitions.archive_draft
_update = transitions.update_section
_usage = transitions.usage_delta
advance_section = policies.advance_section
claim_gate = policies.claim_gate
route_parent = policies.route_parent


def load_chat_model(model_ref: str):
    """Lazy compatibility hook for tests and legacy provider overrides."""
    from ..utils.llm import load_chat_model as load

    return load(model_ref)


async def call_model(system: str, prompt: str, schema=None, *, validator=None, context=None):
    """Compatibility model gateway; new code uses ``agents.bootstrap``."""
    model = load_chat_model(settings.agent.section_model)
    return await invoke_checked(
        model,
        system,
        prompt,
        schema,
        validator=validator,
        context=context,
    )


def resolve_agent_model(spec):
    """Resolve through this facade so existing provider overrides still work."""
    if spec.model_ref != "section_model":
        raise AgentConfigurationError(
            f"Agent '{spec.name}' 使用了未知模型引用：{spec.model_ref}"
        )
    return call_model


async def _retrieve_tool(request):
    return await retrieve_evidence(request)


agent_runner = AgentRunner(
    resolve_agent_model,
    tools={"retrieve_evidence": _retrieve_tool},
    event_sink=dispatch_agent_event,
)


async def plan_sections(state: dict, config: RunnableConfig | None = None) -> dict:
    return await planning_nodes.plan_sections(state, config, runner=agent_runner)


async def research_section(state: dict, config: RunnableConfig | None = None) -> dict:
    return await research_nodes.research_section(state, config, runner=agent_runner)


async def write_section(state: dict, config: RunnableConfig | None = None) -> dict:
    return await writing_nodes.write_section(state, config, runner=agent_runner)


async def review_section(state: dict, config: RunnableConfig | None = None) -> dict:
    return await review_nodes.review_section(state, config, runner=agent_runner)


async def extract_claims(state: dict) -> dict:
    return await review_nodes.extract_claims(state, call_model=call_model)


async def review_report(state: dict, config: RunnableConfig | None = None) -> dict:
    return await report_nodes.review_report(state, config, runner=agent_runner)


async def edit_report(state: dict, config: RunnableConfig | None = None) -> dict:
    return await editorial_nodes.edit_report(state, config, runner=agent_runner)


async def edit_report_section(state: dict, config: RunnableConfig | None = None) -> dict:
    return await editorial_nodes.edit_report_section(state, config, runner=agent_runner)


async def compress_report_section(state: dict, config: RunnableConfig | None = None) -> dict:
    return await editorial_nodes.compress_report_section(state, config, runner=agent_runner)


async def write_report_framing(state: dict, config: RunnableConfig | None = None) -> dict:
    return await editorial_nodes.write_report_framing(state, config, runner=agent_runner)


async def review_edited_report(state: dict, config: RunnableConfig | None = None) -> dict:
    return await editorial_nodes.review_edited_report(state, config, runner=agent_runner)


__all__ = [
    "advance_section",
    "assemble_sections",
    "call_model",
    "claim_gate",
    "compress_report_section",
    "extract_claims",
    "edit_report",
    "edit_report_section",
    "plan_sections",
    "research_section",
    "resolve_agent_model",
    "review_report",
    "review_edited_report",
    "review_section",
    "route_parent",
    "write_section",
    "write_report_framing",
]
