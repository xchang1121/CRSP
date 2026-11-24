#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "$ROOT/scripts/sync_src.sh"

cat > /tmp/launch_exp1.py << 'EOF'
import os, signal, subprocess, time
pid_file = "/content/cspd/exp1.pid"
if os.path.exists(pid_file):
    old_pid = int(open(pid_file).read().strip())
    try:
        os.kill(old_pid, signal.SIGKILL)
        print("Killed old exp1 PID:", old_pid)
        time.sleep(2)
    except ProcessLookupError:
        pass
p = subprocess.Popen(
    ["python3", "-u", "/content/cspd/exp1_reproduce_cspd.py"],
    stdout=open("/content/cspd/exp1.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open(pid_file, "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("Launched fast Stage A exp1_reproduce_cspd.py PID:", p.pid, "poll:", p.poll())
EOF

colab exec -s cspd -f /tmp/launch_exp1.py
