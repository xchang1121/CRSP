import os, signal, subprocess, time
if os.path.exists("/content/cspd/exp10.pid"):
    old = open("/content/cspd/exp10.pid").read().strip()
    if old.isdigit():
        try:
            os.killpg(int(old), signal.SIGKILL)
            print("Killed old exp10 process group", old)
        except Exception as e:
            print("Kill old note:", e)
        time.sleep(2)
os.makedirs("/content/cspd", exist_ok=True)
with open("/content/cspd/exp10.log", "w") as f:
    f.write("launching exp10...\n")
p = subprocess.Popen(
    ["bash", "/content/run_exp10_bg.sh"],
    stdout=open("/content/cspd/exp10.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/cspd/exp10.pid", "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("LAUNCHED exp10 PID=", p.pid, "ALIVE=", os.path.exists(f"/proc/{p.pid}"))
