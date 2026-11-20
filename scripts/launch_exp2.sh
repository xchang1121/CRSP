#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="/mnt/c/Users/singm/.gemini/antigravity/scratch/composable-soft-prompt-distillation"

bash "$ROOT/scripts/sync_src.sh"

cat > /tmp/launch_exp2.py << 'EOF'
import os, signal, subprocess, time
pid_file = "/content/cspd/exp2.pid"
if os.path.exists(pid_file):
    old_pid = int(open(pid_file).read().strip())
    try:
        os.kill(old_pid, signal.SIGKILL)
        print("Killed old exp2 PID:", old_pid)
        time.sleep(2)
    except ProcessLookupError:
        pass
p = subprocess.Popen(
    ["python3", "-u", "/content/cspd/exp2_domain_doc_cl.py"],
    stdout=open("/content/cspd/exp2.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open(pid_file, "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("Launched fast batched exp2_domain_doc_cl.py PID:", p.pid, "poll:", p.poll())
EOF

colab exec -s cspd -f /tmp/launch_exp2.py
