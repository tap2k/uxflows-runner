"""Runtime config — env vars, credential paths, provider knobs.

Single source of truth for "where do credentials live" and "which model".
Phase 0 reads everything from env; later phases will accept overrides per-session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    google_credentials_path: str
    google_project_id: str
    google_location: str
    llm_model: str
    tts_voice: str
    host: str
    port: int

    @classmethod
    def from_env(cls) -> "Config":
        creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not creds:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS not set. Point it at your service-account "
                "JSON (e.g. examples/hello/credentials.json). See README."
            )
        if not Path(creds).is_file():
            raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS path does not exist: {creds}")

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT not set. Use the project ID from your service account."
            )

        return cls(
            google_credentials_path=creds,
            google_project_id=project_id,
            google_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east4"),
            llm_model=os.environ.get("UXFLOWS_LLM_MODEL", "gemini-2.5-flash"),
            tts_voice=os.environ.get("UXFLOWS_TTS_VOICE", "en-US-Chirp3-HD-Charon"),
            host=os.environ.get("UXFLOWS_HOST", "127.0.0.1"),
            port=int(os.environ.get("UXFLOWS_PORT", "8000")),
        )
