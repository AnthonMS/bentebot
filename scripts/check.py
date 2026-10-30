"""Repeatable offline checks. No Discord token, live inference, or deployment is used."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docker", action="store_true", help="Also build the image and run an offline container"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    paths = [str(root / ".test-deps"), str(root)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    checks = [
        (
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "bentebot",
                "bentebot_viewer",
                "bentebot_audio_worker",
                "tests",
                "scripts",
                "deploy/speech",
                "deploy/web-tools",
                "deploy/audio",
                "main.py",
            ],
            60,
        ),
        (
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "bentebot",
                "bentebot_viewer",
                "bentebot_audio_worker",
                "tests",
                "scripts",
                "deploy/speech",
                "deploy/web-tools",
                "deploy/audio",
                "main.py",
            ],
            60,
        ),
        ([sys.executable, "-m", "pytest", "-p", "pytest_asyncio.plugin", "-q"], 180),
        ([sys.executable, "-m", "bentebot", "--check"], 30),
    ]
    if args.docker:
        checks.extend(
            [
                (["docker", "compose", "-f", "deploy/web-tools/compose.yaml", "config", "--quiet"], 30),
                (["docker", "compose", "-f", "deploy/viewer/compose.yaml", "config", "--quiet"], 30),
                (
                    ["docker", "build", "-f", "deploy/viewer/Dockerfile", "-t", "bentebot-viewer:local", "."],
                    600,
                ),
                (["docker", "run", "--rm", "--network", "none", "bentebot-viewer:local", "--check"], 30),
                (["docker", "compose", "--env-file", "example.env", "config", "--quiet"], 30),
                (["docker", "build", "-t", "bentebot:local", "."], 600),
                (
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "bentebot:local",
                        "python",
                        "-m",
                        "bentebot",
                        "--check",
                    ],
                    30,
                ),
            ]
        )
    for command, timeout in checks:
        print(f"Checking: {' '.join(command)}", flush=True)
        try:
            subprocess.run(command, cwd=root, env=env, timeout=timeout, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            print(f"Check failed: {type(exc).__name__}", file=sys.stderr)
            return 1
    print("All requested checks passed. Live Discord/provider quality still requires a call test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
