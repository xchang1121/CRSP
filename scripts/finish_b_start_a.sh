#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Wait cleanly on remote VM until exp2b PID finishes (should be <20s)
cat > /tmp/wait_exp2b.py << 'EOF'
import os, time
pid = open("/content/cspd/exp2b.pid").read().strip()
for _ in range(30):
    if not os.path.exists(f"/proc/{pid}"):
        break
    time.sleep(2)
lines = open("/content/cspd/exp2b.log").readlines()
print("".join(lines[-12:]))
EOF

colab exec -s cspd -f /tmp/wait_exp2b.py
bash "$ROOT/scripts/pull_results.sh" exp2b
bash "$ROOT/scripts/launch_exp1.sh"
