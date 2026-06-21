#!/usr/bin/env python3
"""
Legacy entry point — delegates to cli.py.

Prefer: python cli.py  or  python run_web.py
"""

from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
