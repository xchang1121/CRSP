import os, subprocess, time
os.makedirs("/content/cspd", exist_ok=True)
with open("/content/cspd/exp9b.log", "w") as f:
    f.write("launching exp9b...\n")
p = subprocess.Popen(
    ["bash", "/content/run_exp9b_bg.sh"],
    stdout=open("/content/cspd/exp9b.log", "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
with open("/content/cspd/exp9b.pid", "w") as f:
    f.write(str(p.pid))
time.sleep(2)
print("LAUNCHED exp9b PID=", p.pid, "ALIVE=", os.path.exists(f"/proc/{p.pid}"))
