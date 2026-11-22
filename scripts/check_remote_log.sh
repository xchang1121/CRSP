#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
LOG_NAME="${1:-exp1}"

cat > /tmp/check_log.py << EOF
import os
pid_file = "/content/cspd/${LOG_NAME}.pid"
log_file = "/content/cspd/${LOG_NAME}.log"
pid = open(pid_file).read().strip() if os.path.exists(pid_file) else "none"
alive = pid.isdigit() and os.path.exists(f"/proc/{pid}")
print(f"=== ${LOG_NAME} | PID={pid} | ALIVE={alive} ===")
if os.path.exists(log_file):
    lines = open(log_file).readlines()
    print(f"(total {len(lines)} lines, showing last 45)")
    print("".join(lines[-45:]))
else:
    print("(no log yet)")
EOF

timeout 45 colab exec -s cspd -f /tmp/check_log.py || (timeout 15 colab restart-kernel -s cspd && sleep 8 && timeout 45 colab exec -s cspd -f /tmp/check_log.py)
