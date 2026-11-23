import os, subprocess
pid = open("/content/cspd/exp10b.pid").read().strip() if os.path.exists("/content/cspd/exp10b.pid") else "?"
alive = os.path.exists(f"/proc/{pid}") if pid.isdigit() else False
print(f"PID={pid} ALIVE={alive}")
try:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]).decode().strip()
    print("GPU:", out)
except Exception as e:
    print("nvidia-smi err:", e)
if os.path.exists("/content/cspd/exp10b.log"):
    txt = open("/content/cspd/exp10b.log").read()
    print(f"--- LOG ({len(txt)} chars) ---")
    print(txt[-5000:])
