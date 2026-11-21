import os, subprocess, time
subprocess.run(["pkill", "-9", "-f", "exp8_kv_inversion_and_tf_cr_sp.py"], check=False)
time.sleep(2)
os.makedirs("/content/cspd", exist_ok=True)
log_f = open("/content/cspd/exp8.log", "w")
proc = subprocess.Popen(
    ["bash", "/content/run_exp8_bg.sh"],
    stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True
)
open("/content/cspd/exp8.pid", "w").write(str(proc.pid))
print("KILLED_OLD_AND_STARTED_EXP8_PID:", proc.pid)
