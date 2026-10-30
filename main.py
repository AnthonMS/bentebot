"""Compatibility entry point for the new API-only implementation."""

from bentebot.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
