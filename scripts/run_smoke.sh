#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== ls root ==="
colab ls -s cspd || true

echo "=== mkdir /content/cspd ==="
colab exec -s cspd -f <(echo "import os; os.makedirs('/content/cspd', exist_ok=True); print('cwd=', os.getcwd(), 'ls=', os.listdir('/content'))")

echo "=== upload files ==="
colab upload -s cspd "$ROOT/src/cspd_core.py" cspd/cspd_core.py
colab upload -s cspd "$ROOT/src/benchmarks.py" cspd/benchmarks.py
colab upload -s cspd "$ROOT/src/domain_docs_data.py" cspd/domain_docs_data.py

echo "=== run smoke test ==="
colab exec -s cspd -f "$ROOT/src/smoke_test.py"
