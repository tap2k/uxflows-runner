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
ExitPathType = Literal["happy", "sad", "off", "exit", "return_to_caller"]
CapabilityKind = Literal["retrieval", "function"]
VarType = Literal["string", "number", "boolean", "enum"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lenient(BaseModel):
    """For nodes where the spec carries optional fields we don't yet model
    (e.g. knowledge.tables structure rows). Keep narrow."""

    model_config = ConfigDict(extra="allow")


class Guardrail(Strict):
    id: str
    statement: str


class Capability(Strict):
    id: str
    name: str
    description: str | None = None
    kind: CapabilityKind
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class FAQEntry(Strict):
    question: str
    answer: str
    scripts: dict[str, str] = Field(default_factory=dict)


class GlossaryEntry(Strict):
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
    id: str
    type: ExitPathType
    condition: Condition | None = None  # absent on return_to_caller (per coffee.json)
    next_flow_id: str | None = None
    assigns: dict[str, Assign] = Field(default_factory=dict)
    actions: list[Action] = Field(default_factory=list)


class Routing(Strict):
    entry_condition: Condition | None = None
    exit_paths: list[ExitPath] = Field(default_factory=list)


class Script(Strict):
    id: str
    text: str


class Flow(Strict):
    schema_url: str | None = Field(default=None, alias="$schema")
    id: str
    version: str | None = None
    name: str | None = None
    description: str | None = None
    type: FlowType
    scope: list[str] | None = None
    instructions: str | None = None
    scripts: dict[str, list[Script]] = Field(default_factory=dict)
    guardrails: list[Guardrail] = Field(default_factory=list)
    max_turns: int | None = None
    example: str | None = None
    knowledge: FlowKnowledge | None = None
    variables: dict[str, VariableDecl] = Field(default_factory=dict)
    routing: Routing = Field(default_factory=Routing)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Agent(Strict):
    schema_url: str | None = Field(default=None, alias="$schema")
    id: str
    version: str | None = None
    meta: AgentMeta = Field(default_factory=AgentMeta)
    system_prompt: str | None = None
    chatbot_initiates: bool = False
    guardrails: list[Guardrail] = Field(default_factory=list)
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
