# Translation PoC: LangGraph Fidelity

A bounded experiment to answer one strategic question with evidence rather than theory: **does translation from a UX4 v0 spec to a graph-native runtime (LangGraph) preserve the flow / exit / capture / variable semantics the runner enforces, with enough fidelity to be useful for production deployments?**

This is a spike. Deliverable is *evidence*, not a shippable translator.

## Why

The translation strategy laid out in [`../uxflows/TRANSLATIONS.md`](../uxflows/TRANSLATIONS.md) names three structural classes of export targets. The strategic question — whether to invest meaningfully in graph-native translators as the short-term production path for spec-shaped agents — depends on behavioral fidelity holding in practice, not in theory.

LangGraph is chosen for this PoC because:

- Its structural mapping to UX4 is the cleanest of any target (graph nodes, typed state, conditional edges, native interrupts).
- Text-mode validation is faster to instrument than voice. Text mode is already first-class in the runner ([`server/text_session.py`](src/uxflows_runner/server/text_session.py)).
- Lower implementation risk than Pipecat (which reintroduces tool-call atomicity concerns in the routing decision path).

If LangGraph fidelity holds, the same experiment shape rolls forward to Pipecat on a hardened IR. If it doesn't, that's strong evidence to weight the runner-as-production path over translation-as-production.

## What decision this gates

After the PoC:

- **Fidelity holds** → invest in generalizing the LangGraph translator and formalizing the harness; begin Pipecat translator on the same IR. Translation becomes the short-term production path per the strategy in TRANSLATIONS.md.
- **Fidelity diverges on identifiable categories** → named gaps to address before generalizing; smaller targeted investment to close them.
- **Fidelity fundamentally breaks** → reweight toward runner-as-production-runtime. Translators become escape-hatch for specific customer deals, not the strategic production lever.

The PoC is falsifiable. The outcome should shape next-quarter investment; if it wouldn't, don't run the experiment.

## Scope

**In:**

- FNOL spec only (small, real, currently being authored against).
- Text mode only.
- LangGraph target only.
- Translator coverage limited to the spec features FNOL actually uses.

**Spec features covered:**

- `variables` with type annotations → Pydantic state schema.
- `capabilities` of `kind: function` over HTTP → `@tool` functions.
- `flows` with entry/exit semantics → graph nodes.
- `exit_paths` with `calculation` conditions → conditional edges via ported expression eval.
- `exit_paths` with `llm` conditions → conditional edges via LLM-judgment helper.
- `exit_path.actions` → capability invocation with output binding to state (per the capability-output binding decision in [RUNNER-PLAN.md](RUNNER-PLAN.md)).
- `entry_condition` on flows → guards before node entry.
- Per-flow system prompts composed from spec scripts + persona.
- `agent.chatbot_initiates` → entry node sends opening turn.

**Out of scope:**

- Multilingual (one language only).
- Interrupts (FNOL doesn't use them; revisit if a later spec requires them).
- Knowledge.faq / knowledge.tables (unless FNOL uses them).
- Voice-specific anything.
- MCP capabilities (HTTP only).
- Productionization of the translator — this is a spike.
- Editor integration.

## Architecture

```
uxflows-runner/
  experiments/langgraph_poc/
    __init__.py
    translator.py         # UX4 spec → Python source
    runtime_helpers.py    # Shared expression eval, capability mock, LLM helper
    generated/
      fnol.py             # Translator output
    test_fidelity.py      # Harness + scenarios
    fixtures/
      capabilities.json   # Shared mock returns for runner + translated
```

Why this location: the harness needs runner internals to play scenarios through `TextSession`, so the code lives next to the runner. If the translator productizes after the PoC, it moves to `../uxflows/lib/codegen/langgraph/` to join the editor's codegen pipeline.

### Generated artifact shape

```python
# generated/fnol.py
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from pydantic import BaseModel
from runtime_helpers import eval_expr, call_capability, llm_turn

class FNOLState(BaseModel):
    policy_number: str | None = None
    policy_active: bool | None = None
    # ... derived from spec.variables

@tool
async def verify_policy(policy_number: str) -> dict:
    return await call_capability("verify_policy", {"policy_number": policy_number})

INTAKE_PROMPT = """..."""   # composed from spec scripts + persona

async def flow_intake(state: FNOLState) -> dict: ...
async def flow_verify(state: FNOLState) -> dict:
    result = await verify_policy.ainvoke({"policy_number": state.policy_number})
    return {"policy_active": result["policy_active"]}

def route_after_verify(state: FNOLState) -> str:
    if eval_expr("policy_active != True", state): return "policy_invalid"
    return "claim_details"

graph = StateGraph(FNOLState)
# ... add nodes, edges
app = graph.compile(checkpointer=MemorySaver())
```

Conversational pause for user input uses LangGraph's `interrupt()` + checkpointer pattern. Each flow node either yields-for-user (interrupt) or routes deterministically (when exit_path conditions resolve without needing a turn).

### Test harness shape

```python
# test_fidelity.py
SCENARIOS = [
    {
        "name": "happy_path",
        "fixtures": {"verify_policy": {"POL-123": {"policy_active": True}}, ...},
        "user_turns": ["I need to file a claim", "POL-123", ...],
        "expect": {
            "flow_path": ["intake", "verify", "claim_details", "file", "confirm"],
            "final_vars": {"policy_active": True, "claim_id": "CLM-001"},
            "final_exit": "confirm_filed",
        },
    },
    # 3-4 more
]

async def run_runner(spec, scenario): ...    # TextSession with mock capabilities
async def run_translated(scenario): ...      # compiled LangGraph app, same fixtures
def assert_fidelity(runner_trace, translated_trace, expected): ...
```

## Equivalence criteria

The PoC validates **dispatch logic**, not wording.

- ✅ Same flow transition sequence — both surfaces visit the same flows in the same order.
- ✅ Same final variable state — every variable referenced in the scenario has the same value at the end.
- ✅ Same final exit decision — the terminal exit path matches.
- ❌ Same per-turn text — out of scope. LLM nondeterminism defeats text-level comparison even at temperature 0, and wording isn't what translation is supposed to preserve. Phrasing fidelity is a separate concern.

## Scenarios

Minimum four; an optional fifth if time allows.

1. **Happy path** — full FNOL flow with valid policy, claim filed successfully.
2. **Invalid policy** — `verify_policy` returns `{policy_active: False}`; route through the invalid-policy branch.
3. **Capability failure** — `verify_policy` raises; `policy_active` stays undefined; downstream `var != True` branches fire correctly. Validates the "Failure → undefined" semantic from the [capability-output binding decision](RUNNER-PLAN.md).
4. **Missing capture** — user gives incomplete info mid-flow; flow re-prompts; eventually captures and proceeds. Validates capture loop behavior.
5. **(Optional) Calculation branch** — exit path with a non-trivial expression (date comparison, multi-variable condition). Validates expression eval port.

## Effort & sequencing

| Day | Work |
|---|---|
| 1 | Pick FNOL spec; hand-translate to LangGraph on paper. Validate every spec feature has a target equivalent. Smoke out gaps before writing code. |
| 2 | Wire `runtime_helpers`: expression eval (port from [`expressions.py`](src/uxflows_runner/dispatcher/expressions.py)), `call_capability` with fixture lookup, `llm_turn` / `llm_judge` helpers. |
| 3-4 | Build translator: walk `LoadedSpec`, emit Python source for state schema, tools, flow nodes, conditional edges, graph construction. |
| 5 | First end-to-end run of generated `fnol.py` against a hardcoded scenario. One happy path green. |
| 6 | Test harness: scenario runner for both surfaces, fixture injection, trace comparison helpers. |
| 7-8 | Write 3-5 scenarios; run both surfaces; debug divergences. Most likely failure modes: condition expression semantics, capture extraction differences, output binding timing. |
| 9-10 | Iterate to green or characterize the divergence cleanly. |
| 11-12 | Write up findings: what passed, what diverged, what the divergences teach us. |

~10-12 focused days, ~2-3 calendar weeks.

## Upfront decisions

1. **LangGraph version** — pin specifically in `pyproject.toml`. The ecosystem moves fast; reproducibility matters.
2. **Conversational pause mechanism** — `interrupt()` + `MemorySaver` checkpointer. 30-min spike at day 1 before committing to the design.
3. **Capability mock injection** — both surfaces read the same `fixtures/capabilities.json`. The runner gets a `MockCapabilityDispatcher` injected via existing config seams; the translated runtime's `call_capability` reads the same file.
4. **LLM provider** — same as runner (Vertex Gemini), via LangChain's `ChatVertexAI`. Same temperature (0). Same credentials. Maximum reproducibility.
5. **Expression eval handling** — copy the runner's logic from [`expressions.py`](src/uxflows_runner/dispatcher/expressions.py) into `runtime_helpers` for the PoC. Don't share Python modules across surfaces yet; productize later if the PoC succeeds.

## Risks

- **Prompt fidelity divergence.** Runner's prompt assembly and translated agent's per-flow prompts will differ in structure. Even at temp 0 with the same model, decisions may drift. Mitigation: align prompts as closely as practical; equivalence target is "same dispatch decisions," not "same prompts."
- **Conversational state mechanics.** LangGraph's `interrupt()` + checkpointer pattern has its own quirks. Mitigation: 30-min spike before committing; budget 1 day buffer.
- **Mock backend consistency.** Drift in fixture-loading code between runner and translated surfaces would cause boring divergences. Mitigation: shared `capabilities.json` schema and shared loader.
- **LLM nondeterminism even at temp 0.** Mitigation: equivalence criteria don't compare text; run each scenario multiple times and require consistent dispatch decisions across runs.
- **Spec gaps surfaced mid-PoC.** FNOL may exercise a feature the translator design doesn't handle. Mitigation: day-1 hand-translation flushes most of these; budget 1 day buffer for the rest.
- **Information value depreciation.** If the next quarter's plan doesn't change based on the outcome, the PoC is academic. Mitigation: commit to the go/no-go below before starting.

## Go / no-go criteria (post-PoC)

**Success looks like:** all 4 scenarios pass equivalence on flow path + final vars + final exit. Translator covers all FNOL features. Divergences, if any, are explainable and addressable.

→ Generalize the translator to cover the full schema. Formalize the harness (golden trace generation, scenario authoring tooling). Begin Pipecat translator on the same IR. Update [`../uxflows/TRANSLATIONS.md`](../uxflows/TRANSLATIONS.md) with concrete fidelity results.

**Partial success looks like:** scenarios diverge on identifiable categories (e.g., `llm` conditions drift, `calculation` conditions hold). Specific gaps named.

→ Address the named gaps before generalizing. Reassess whether they're addressable in the translator or indicate a structural limit of LangGraph-as-target.

**Failure looks like:** scenarios diverge unpredictably with no clear pattern, or diverge in ways that require fundamental reshaping of either the translator or the runner.

→ Reweight toward runner-as-production. Per [TRANSLATIONS.md](../uxflows/TRANSLATIONS.md), the runner already absorbs Pipecat for voice; growing it for production becomes the dominant path. Translators stay escape-hatch for specific deals. Document what failed and why — the negative result is itself valuable evidence.
