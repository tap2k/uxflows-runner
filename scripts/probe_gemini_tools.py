"""Throwaway probe — issue ONE Vertex Gemini call with multiple function-tool
variants modeled like the dispatcher's per-turn call shape, dump the raw
response so we can confirm the tool-call format before building prompt_builder.

What we want to learn:
  1. Does Gemini emit assistant TEXT and a tool CALL in the same response, or
     does emitting a tool call suppress the natural-language reply? (The
     "one LLM call per turn" rule depends on getting both.)
  2. What does the function_call payload look like? (parts[].function_call vs.
     a top-level field, arg shape.)
  3. Can we force / bias the model to *always* emit a tool call (function-calling
     mode = ANY) so the dispatcher always knows the routing decision?
  4. Are enum-typed parameters honored?

Run with:
  uv run python scripts/probe_gemini_tools.py
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from google.genai import Client, types
from google.oauth2 import service_account

load_dotenv()

CREDS_PATH = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4")
MODEL = os.environ.get("UXFLOWS_LLM_MODEL", "gemini-2.5-flash")

credentials = service_account.Credentials.from_service_account_file(
    CREDS_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
client = Client(vertexai=True, project=PROJECT, location=LOCATION, credentials=credentials)


# Modeled like the per-turn schema for flow_greet from coffee.json:
# - take_exit_path with a routing decision
# - trigger_interrupt with the matched interrupt id
TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="take_exit_path",
                description=(
                    "Pick an exit path for the current flow. Use this when the patron "
                    "has revealed enough to route — coffee, tea, or walking away."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "exit_path_id": types.Schema(
                            type=types.Type.STRING,
                            enum=["xp_greet_to_coffee", "xp_greet_to_tea", "xp_greet_walkaway"],
                            description="Which exit path to take.",
                        ),
                        "drink_type": types.Schema(
                            type=types.Type.STRING,
                            enum=["coffee", "tea"],
                            description=(
                                "Required for xp_greet_to_coffee or xp_greet_to_tea. "
                                "Omit for xp_greet_walkaway."
                            ),
                        ),
                    },
                    required=["exit_path_id"],
                ),
            ),
            types.FunctionDeclaration(
                name="trigger_interrupt",
                description=(
                    "Trigger an interrupt flow because the patron asked something "
                    "off the routing path."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "interrupt_flow_id": types.Schema(
                            type=types.Type.STRING,
                            enum=["int_menu"],
                            description="Which interrupt to trigger.",
                        ),
                    },
                    required=["interrupt_flow_id"],
                ),
            ),
        ]
    )
]

SYSTEM = (
    "You are Skye, a friendly barista at Bluebird Coffee. Greet the patron and "
    "figure out coffee or tea. Reply naturally in one short sentence AND, on the "
    "same turn, call exactly one tool: take_exit_path when you can route, or "
    "trigger_interrupt when the patron asks something off-path."
)

CASES = [
    ("clear-coffee", "I'd love a latte please"),
    ("clear-tea", "could I get a green tea?"),
    ("ambiguous", "hi, what do you have?"),
    ("walkaway", "actually never mind, I'll come back later"),
]


def run(label: str, user_text: str, tool_mode: str) -> None:
    print(f"\n=== {label} | mode={tool_mode} | user={user_text!r} ===")
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        tools=TOOLS,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=tool_mode)
        ),
        temperature=0.2,
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=[types.Content(role="user", parts=[types.Part(text=user_text)])],
        config=config,
    )
    cand = resp.candidates[0]
    print(f"finish_reason: {cand.finish_reason}")
    for i, part in enumerate(cand.content.parts or []):
        if part.text:
            print(f"  part[{i}].text: {part.text!r}")
        if part.function_call:
            fc = part.function_call
            print(f"  part[{i}].function_call: name={fc.name} args={json.dumps(dict(fc.args))}")


def main() -> None:
    # AUTO = model decides whether to call a tool. ANY = forced to call.
    for mode in ("AUTO", "ANY"):
        for label, user_text in CASES:
            run(label, user_text, mode)


if __name__ == "__main__":
    main()
