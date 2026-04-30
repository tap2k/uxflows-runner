"""Command-line entry point.

Usage:
    uxflows-runner serve            # start the FastAPI server
"""

from __future__ import annotations

import argparse

import uvicorn

from uxflows_runner.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(prog="uxflows-runner")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="Start the FastAPI server")
    args = parser.parse_args()

    if args.cmd == "serve":
        config = Config.from_env()
        uvicorn.run(
            "uxflows_runner.server.app:app",
            host=config.host,
            port=config.port,
            reload=False,
        )


if __name__ == "__main__":
    main()
