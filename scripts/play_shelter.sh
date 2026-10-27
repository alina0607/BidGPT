#!/bin/zsh
# Starts the ML server and the game together on the same port (owner, 2026-10-27: port 5555 is taken by Docker on
# this Mac, and a game started without its server cannot paint the shelter). The server stops when the game closes.
#   scripts/play_shelter.sh            # port 5577
#   scripts/play_shelter.sh 5600       # another port
ROOT="${0:A:h:h}"
PORT="${1:-5577}"
LOG="$ROOT/ml/logs/server_$PORT.log"
mkdir -p "$ROOT/ml/logs"
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $PORT is already in use; using the server that is listening there"
else
  (cd "$ROOT/ml" && python3 -m sellanything_ml.server --port "$PORT" > "$LOG" 2>&1) &
  SERVER=$!
  trap 'kill $SERVER 2>/dev/null' EXIT
  for i in {1..60}; do grep -q "listening" "$LOG" 2>/dev/null && break; sleep 0.5; done
  grep -q "listening" "$LOG" || { echo "the server did not start; see $LOG"; exit 1; }
  echo "server on 127.0.0.1:$PORT (log: $LOG)"
fi
"$ROOT/build/game/sell_anything" --server "127.0.0.1:$PORT"
