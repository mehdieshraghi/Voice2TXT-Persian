#!/usr/bin/env python3
"""Run the Flask web UI."""

import argparse
import logging

from app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice2TXT Persian — Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    parser.add_argument("--settings", default=None, help="Path to settings JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    app = create_app(settings_path=args.settings)
    print(f"Open http://{args.host}:{args.port} in your browser")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
