#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT/scripts/sync_src.sh"

cat > /tmp/launch_exp2b.py << 'EOF'
import subprocess, time
p = subprocess.Popen(
    ["python3", "-u", "/content/cspd/exp2b_doc_nll_cl.py"],
    stdout=open("/content/cspd/exp2b.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/cspd/exp2b.pid", "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("Launched exp2b_doc_nll_cl.py PID:", p.pid, "poll:", p.poll())
EOF

colab exec -s cspd -f /tmp/launch_exp2b.py
