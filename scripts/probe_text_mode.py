"""Probe — Path A viability for SIMULATE-PLAN.md text adapter.

Question to answer: can we drive `GoogleLLMService` (the AI Studio variant,
not Vertex) standalone — outside any Pipecat pipeline — using a plain
`GOOGLE_API_KEY`, against a real `LLMContext` carrying the kind of system
prompt + tool schema the dispatcher emits per turn?

If yes, the text adapter (`server/text_session.py`) reuses the existing tool
handlers in `dispatcher/processor.py` mostly as-is. If no — if construction
needs monkey-patching or `LLMContext` won't build standalone — we fall back
to Path B (call `google.genai` directly + reimplement decision application).

Probe phases:
  1. CONSTRUCTION (no network) — instantiate `GoogleLLMService(api_key="...")`,
     instantiate `LLMContext` standalone with messages + tools, confirm the
     adapter can pull invocation params out of it. This alone validates the
     wiring boundary even without a real key.
  2. INFERENCE (only if GOOGLE_API_KEY is set) — issue ONE real call against
     gemini-2.5-flash with coffee.json-shaped tools. Confirm the response
     contains both an assistant TEXT part and a `take_exit_path` function_call,
     mirroring probe_gemini_tools.py's findings but through the Pipecat surface.

Run with:
  uv run python scripts/probe_text_mode.py
  GOOGLE_API_KEY=... uv run python scripts/probe_text_mode.py   # full probe
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv

load_dotenv()


def phase1_construction() -> "tuple[object, object]":
    """Build GoogleLLMService + LLMContext standalone. No network."""
    print("=" * 72)
    print("PHASE 1 — construction (no network)")
    print("=" * 72)

    from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.google.llm import GoogleLLMService

    api_key = os.environ.get("GOOGLE_API_KEY", "dummy-for-construction-probe")

    print(f"  instantiating GoogleLLMService(api_key={'<real>' if api_key != 'dummy-for-construction-probe' else '<dummy>'})...")
    llm = GoogleLLMService(
        api_key=api_key,
        settings=GoogleLLMService.Settings(model="gemini-2.5-flash"),
    )
    print(f"    OK — type={type(llm).__name__}, model={llm._settings.model}")
    print(f"    client constructed: {llm._client is not None}")

    # Coffee.json-shaped tools, lifted from probe_gemini_tools.py but as a
    # ToolsSchema (Pipecat's universal type) so we exercise the adapter
    # translation the dispatcher actually uses.
    from pipecat.adapters.schemas.function_schema import FunctionSchema
    from pipecat.adapters.schemas.tools_schema import ToolsSchema

    take_exit_path = FunctionSchema(
        name="take_exit_path",
        description="Route out of flow_greet onto a declared exit path. Call on the same turn as your spoken reply.",
        properties={
            "exit_path_id": {
                "type": "string",
                "enum": ["xp_greet_to_coffee", "xp_greet_to_tea", "xp_greet_walkaway"],
                "description": "Pick the exit path matching the patron's current state.",
            },
            "drink_type": {
                "type": "string",
                "description": "Captured drink type (coffee/tea variants). Omit on walkaway.",
            },
        },
        required=["exit_path_id"],
    )
    trigger_interrupt = FunctionSchema(
        name="trigger_interrupt",
        description="Trigger an interrupt flow when the patron asks an off-path question.",
        properties={
            "interrupt_flow_id": {
                "type": "string",
                "enum": ["int_menu"],
                "description": "Which interrupt flow to enter.",
            },
        },
        required=["interrupt_flow_id"],
    )
    tools_schema = ToolsSchema(standard_tools=[take_exit_path, trigger_interrupt])

    print("  building LLMContext standalone with system + user message + tools...")
    context = LLMContext(
        messages=[
            {"role": "system", "content": "You are a barista taking coffee orders. Reply briefly."},
            {"role": "user", "content": "I'd love a latte."},
        ],
        tools=tools_schema,
    )
    print(f"    OK — context.messages count={len(context.messages)}, tools attached={context.tools is not None}")

    # Verify the adapter can produce invocation params from this context the
    # same way the LLM service does internally during inference.
    print("  invoking GeminiLLMAdapter.get_llm_invocation_params(context)...")
    adapter = GeminiLLMAdapter()
    params = adapter.get_llm_invocation_params(context)
    print(f"    OK — keys={sorted(params.keys())}")
    print(f"    messages count={len(params['messages'])}")
    print(f"    system_instruction set: {bool(params.get('system_instruction'))}")
    print(f"    tools count={len(params['tools']) if params.get('tools') else 0}")

    # Sanity: the tools list should now be Gemini-native function declarations
    # (not our ToolsSchema). That confirms the translation we'd otherwise have
    # to reimplement in Path B is happening for free.
    if params.get("tools"):
        first_tool = params["tools"][0]
        print(f"    first tool type: {type(first_tool).__name__}")
        # google.genai.types.Tool — has function_declarations
        if hasattr(first_tool, "function_declarations"):
            decls = first_tool.function_declarations
            print(f"    function_declarations count={len(decls)}")
            print(f"    first declaration name={decls[0].name}")

    return llm, context


async def phase2_inference(llm, context) -> None:
    """Make ONE real Gemini call. Confirms Path A is end-to-end live."""
    print()
    print("=" * 72)
    print("PHASE 2 — live inference")
    print("=" * 72)

    if os.environ.get("GOOGLE_API_KEY") is None:
        print("  GOOGLE_API_KEY not set — skipping (Phase 1 alone proves construction).")
        print("  To run Phase 2: GOOGLE_API_KEY=... uv run python scripts/probe_text_mode.py")
        return

    # Mirror what GoogleLLMService.run_inference does, but read BOTH text and
    # function_call parts (run_inference returns text only). This is the
    # exact pattern TextSession.turn() will use.
    from google.genai.types import GenerateContentConfig

    from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter

    adapter = GeminiLLMAdapter()
    params = adapter.get_llm_invocation_params(context)

    generation_params = llm._build_generation_params(
        system_instruction=params["system_instruction"],
        tools=params["tools"] if params["tools"] else None,
    )
    # gemini-2.5-flash defaults to dynamic thinking — disable for low TTFT,
    # matching what _stream_content does in the pipeline path.
    generation_params["thinking_config"] = {"thinking_budget": 0}

    print(f"  calling generate_content (model={llm._settings.model})...")
    response = await llm._client.aio.models.generate_content(
        model=llm._settings.model,
        contents=params["messages"],
        config=GenerateContentConfig(**generation_params),
    )

    print()
    print("  --- response inspection ---")
    if not response.candidates:
        print("  ERROR — no candidates returned")
        return

    cand = response.candidates[0]
    if not cand.content or not cand.content.parts:
        print("  ERROR — candidate has no content/parts")
        return

    text_parts: list[str] = []
    function_calls: list[tuple[str, dict]] = []
    for i, part in enumerate(cand.content.parts):
        if part.text:
            text_parts.append(part.text)
            print(f"  part[{i}] TEXT: {part.text!r}")
        if part.function_call:
            fc = part.function_call
            args = dict(fc.args) if fc.args else {}
            function_calls.append((fc.name, args))
            print(f"  part[{i}] CALL: {fc.name}({json.dumps(args)})")

    print()
    print("  --- verdict ---")
    print(f"  text parts: {len(text_parts)}")
    print(f"  function calls: {len(function_calls)}")
    if text_parts and function_calls:
        print("  ✅ BOTH text and function_call returned in one response — same as voice path.")
    elif function_calls and not text_parts:
        print("  ⚠️  function_call returned WITHOUT text — would need a follow-up inference.")
    elif text_parts and not function_calls:
        print("  ⚠️  text returned WITHOUT function_call — model didn't route on this input.")
    else:
        print("  ❌ neither text nor function_call returned.")


async def main() -> None:
    llm, context = phase1_construction()
    await phase2_inference(llm, context)
    print()
    print("=" * 72)
    print("DONE — see verdict above. Path A is viable iff Phase 1 completed cleanly")
    print("and Phase 2 (when run with a key) returned text+call together.")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())
