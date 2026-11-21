#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

cat > /tmp/check_kv_api.py << 'EOF'
import inspect, subprocess, sys, os, torch, transformers
from transformers.cache_utils import DynamicCache
print('torch:', torch.__version__, '| transformers:', transformers.__version__, '| GPU:', torch.cuda.get_device_name(0))
print('DynamicCache.update sig:', inspect.signature(DynamicCache.update))
print('DynamicCache.get_seq_length src:\n', inspect.getsource(DynamicCache.get_seq_length))
print('DynamicCache.update src:\n', inspect.getsource(DynamicCache.update))
if not os.path.exists('/content/IFBench'):
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/allenai/IFBench.git', '/content/IFBench'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'langdetect', 'immutabledict', 'emoji', 'syllapy'], check=True)
import nltk
for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'stopwords']:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass
print('IFBench ready!')
EOF

timeout 60 colab exec -s cspd -f /tmp/check_kv_api.py || (timeout 15 colab restart-kernel -s cspd && sleep 1 && timeout 60 colab exec -s cspd -f /tmp/check_kv_api.py)
