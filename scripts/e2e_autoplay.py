"""End-to-end test: the real game plays one round against the real ML server, headless.

    python scripts/e2e_autoplay.py [--game build/game/sell_anything] [--item bicycle]

Starts the server on a free port with a fixed sketch label and phrase-pool buyers (no trained
model, no torch), runs the game with the null backend and --autoplay (draw → confirm → answer
every question → sell), and passes only if the game loaded its buyer art and reports a sale. On
failure it prints the game's transcript and the server's log, which is usually enough to see which
step broke.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME_BUDGET = 20_000          # null-backend frames are ~1 ms: a 20 s ceiling for the whole round


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, server: subprocess.Popen, timeout: float = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError("server exited before it started listening")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1) as s:
                s.sendall(b'{"type": "ping"}\n')
                if b"pong" in s.recv(1024):
                    return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"server did not answer ping within {timeout:.0f}s")


def find_game() -> Path:
    for candidate in ("build/game/sell_anything", "build/game/Release/sell_anything.exe",
                      "build/game/sell_anything.exe", "build/game/Debug/sell_anything.exe"):
        if (ROOT / candidate).exists():
            return ROOT / candidate
    raise FileNotFoundError("game binary not found; build first: cmake -S . -B build && cmake --build build")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", type=Path, help="path to the sell_anything binary")
    parser.add_argument("--item", default="bicycle", help="what the fixed sketch classifier answers")
    args = parser.parse_args()
    game = args.game or find_game()

    port = free_port()
    server_log = ROOT / "build" / "e2e_server.log"
    with open(server_log, "w") as log:
        server = subprocess.Popen(
            [sys.executable, "-m", "sellanything_ml.server", "--port", str(port),
             "--fixed-sketch", args.item, "--phrases-only"],
            cwd=ROOT / "ml", stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_for_server(port, server)
        started = time.time()
        run = subprocess.run(
            [str(game), "--backend", "null", "--autoplay", "--server", f"127.0.0.1:{port}"],
            capture_output=True, text=True, timeout=120, env={**os.environ, "SA_EXIT_AFTER": str(FRAME_BUDGET)})
        seconds = time.time() - started
    finally:
        server.terminate()
        server.wait(timeout=10)

    transcript = run.stdout.strip().splitlines()
    # The game reports how many buyer looks it loaded; none means missing or unreadable art.
    art_loaded = any(line.startswith("art: ") and not line.startswith("art: 0 ") for line in transcript)
    # Autoplay types every other answer; at least one has to be understood as an option.
    typed_ok = any(line.startswith('heard: {') for line in transcript)
    ok = run.returncode == 0 and art_loaded and typed_ok and any(" buys your " in line for line in transcript)
    print("\n".join(transcript))
    if run.stderr.strip():
        print("--- game stderr ---\n" + run.stderr.strip())
    if not ok:
        print(f"--- server log ({server_log}) ---\n" + server_log.read_text().strip())
    print(f"e2e: {'PASS' if ok else 'FAIL'} (exit {run.returncode}, {len(transcript)} lines, {seconds:.1f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
