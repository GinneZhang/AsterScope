#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import socket
import time

targets = [
    ("127.0.0.1", 5432, "postgres"),
    ("127.0.0.1", 6379, "redis"),
    ("127.0.0.1", 7687, "neo4j"),
    ("127.0.0.1", 9200, "elasticsearch"),
]

deadline = time.time() + 180

for host, port, name in targets:
    while time.time() < deadline:
        sock = socket.socket()
        sock.settimeout(1.5)
        try:
            sock.connect((host, port))
            sock.close()
            print(f"{name} ready on {host}:{port}")
            break
        except OSError:
            time.sleep(2)
        finally:
            try:
                sock.close()
            except OSError:
                pass
    else:
        raise SystemExit(f"Timed out waiting for {name} on {host}:{port}")
PY
