import os
import subprocess
import sys
import time

COLAB = "/home/singm/.local/bin/colab"


def get_env():
    env = dict(os.environ)
    # Use default WSL route only if user proxy listens on gateway; fall back to direct env
    if os.environ.get("USE_GW_PROXY") == "1":
        gw = subprocess.check_output("ip route | awk '/default/{print $3}'", shell=True, text=True).strip()
        if gw:
            env["HTTP_PROXY"] = f"http://{gw}:7890"
            env["HTTPS_PROXY"] = f"http://{gw}:7890"
    return env


def run_colab(args, retries=6, timeout=30):
    env = get_env()
    for i in range(retries):
        try:
            r = subprocess.run(
                [COLAB] + args,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            combined = (r.stdout or "") + (r.stderr or "")
            if r.returncode == 0 and "Download failed:" not in combined and "[colab] Error:" not in combined:
                return r.returncode, r.stdout, r.stderr
            if any(k in combined for k in ("SSLEOFError", "SSLError", "ConnectTimeoutError", "ConnectionError", "Connection reset", "WebSocketConnectionClosedException")):
                time.sleep(2 + i)
                continue
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            time.sleep(2 + i)
    return 1, "", "Exceeded retries"


if __name__ == "__main__":
    rc, out, err = run_colab(sys.argv[1:])
    if out:
        sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    sys.exit(rc)
