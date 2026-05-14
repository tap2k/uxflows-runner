"""Pydantic models mirroring the v0 schema in ../uxflows/SCHEMA.md.

Hand-mirrored — drift risk is real (see RUNNER-PLAN.md §Risks). When the schema
changes, update here in lockstep. Validation against examples/coffee.json runs
in tests/test_spec_loader.py.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Method = Literal["llm", "calculation", "direct"]
FlowType = Literal["happy", "sad", "off", "utility", "interrupt"]
CapabilityKind = Literal["retrieval", "function"]
VarType = Literal["string", "number", "boolean", "enum"]

# A translatable string. Either a plain string (monolingual; the value is in
# the agent's default language) or a Record keyed by language code. The runner
# resolves to a single string via `resolve_localized`.
LocalizedString = str | dict[str, str]

# Reserved `goto` keywords. Anything else is treated as a flow id reference.
GOTO_END = "END"
GOTO_RETURN = "RETURN"


def is_end_goto(goto: str) -> bool:
    return goto == GOTO_END


def is_return_goto(goto: str) -> bool:
    return goto == GOTO_RETURN


def is_flow_goto(goto: str) -> bool:
    return goto not in (GOTO_END, GOTO_RETURN)


def default_language(languages: list[str] | None) -> str:
    """First entry of `agent.meta.languages` is the default; fall back to "EN"
    if languages is missing or empty (legacy / pre-multilingual specs)."""
    if not languages:
        return "EN"
    return languages[0]


def resolve_localized(
    value: LocalizedString | None,
    lang: str | None,
    default_lang: str,
) -> str:
    """Resolve a LocalizedString to a single string. Fallback order:
    requested lang → default lang → any value present → "".
    `lang=None` is treated as the default language."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    effective_lang = lang or default_lang
    if effective_lang in value:
        return value[effective_lang]
    if default_lang in value:
        return value[default_lang]
    for key in value:
        return value[key]
    return ""


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lenient(BaseModel):
    """For nodes where the spec carries optional fields we don't yet model
    (e.g. knowledge.tables structure rows). Keep narrow."""

    model_config = ConfigDict(extra="allow")


class Guardrail(Strict):
    id: str
    statement: str


class BusinessGoal(Strict):
    id: str
    name: str
    expression: str
    method: Method


class Capability(Strict):
    id: str
    name: str
    description: str | None = None
    kind: CapabilityKind
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class FAQEntry(Strict):
    id: str
    question: str
    answer: LocalizedString


class GlossaryEntry(Strict):
    id: str
    term: str
    definition: str


class KnowledgeTableField(Strict):
    field: str
    description: str | None = None
    type: str | None = None


class KnowledgeTable(Lenient):
    id: str
    name: str
    purpose: str | None = None
    structure: list[KnowledgeTableField] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    scaling_rule: str | None = None


class AgentKnowledge(Strict):
    faq: list[FAQEntry] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    tables: list[KnowledgeTable] = Field(default_factory=list)


class FlowKnowledge(Strict):
    faq: list[FAQEntry] = Field(default_factory=list)


class VariableDecl(Strict):
    type: VarType | None = None
    description: str | None = None
    values: list[Any] | None = None


class AgentMeta(Strict):
    name: str | None = None
    purpose: str | None = None
    client: str | None = None
    languages: list[str] = Field(default_factory=list)
    modes: list[Literal["voice", "text"]] = Field(default_factory=list)


class Condition(Strict):
    expression: str
    method: Method
    pattern: str | None = None  # used when method == "calculation" for regex subtype


class Assign(Strict):
    method: Method
    value: Any = None  # required for method=="direct"; ignored for "llm"
    pattern: str | None = None  # for "calculation" pattern-match subtype


class Action(Strict):
    capability_id: str


class ExitPath(Strict):
    """An edge out of a flow.

    `goto` is either a flow id, or one of the reserved keywords:
      - "END"     — terminate the conversation
      - "RETURN"  — pop the call frame and resume the caller (or END at top level)

    The old `type` (happy/sad/off/exit/return_to_caller) and `next_flow_id`
    fields are gone — destinations come from `goto`, and tone labels derive
    from the destination flow's type when needed.
    """

    id: str
    goto: str
    condition: Condition | None = None
    notes: str | None = None  # authoring annotation; ignored by runtime
    assigns: dict[str, Assign] = Field(default_factory=dict)
    actions: list[Action] = Field(default_factory=list)


class Script(Strict):
    """A scripted line for one flow. `text` is a LocalizedString; `variations`
    are per-language alternative phrasings."""

    id: str
    text: LocalizedString
    variations: dict[str, list[str]] | None = None


class Flow(Strict):
    schema_url: str | None = Field(default=None, alias="$schema")
    id: str
    version: str | None = None
    name: str | None = None
    type: FlowType
    instructions: str | None = None
    entry_condition: Condition | None = None
    exit_paths: list[ExitPath] = Field(default_factory=list)
    scripts: list[Script] = Field(default_factory=list)
    guardrails: list[Guardrail] = Field(default_factory=list)
    notes: str | None = None  # authoring annotation; ignored by runtime
    example: str | None = None
    knowledge: FlowKnowledge | None = None
    variables: dict[str, VariableDecl] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @property
    def is_callable(self) -> bool:
        """A flow is callable iff at least one of its exit paths returns to
        the caller. Entering a callable flow pushes a call frame; taking a
        RETURN exit pops it."""
        return any(is_return_goto(ep.goto) for ep in self.exit_paths)


class Agent(Strict):
    schema_url: str | None = Field(default=None, alias="$schema")
    id: str
    version: str | None = None
    meta: AgentMeta = Field(default_factory=AgentMeta)
    system_prompt: str | None = None
    chatbot_initiates: bool = False
    guardrails: list[Guardrail] = Field(default_factory=list)
    business_goals: list[BusinessGoal] = Field(default_factory=list)
    capabilities: list[Capability] = Field(default_factory=list)
    knowledge: AgentKnowledge = Field(default_factory=AgentKnowledge)
    variables: dict[str, VariableDecl] = Field(default_factory=dict)
    entry_flow_id: str

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Spec(Strict):
    """Top-level spec doc. Coffee fixture wraps {agent, flows[]}; the schema
    doc shows them as separate snippets but they ship together."""

    agent: Agent
    flows: list[Flow]
