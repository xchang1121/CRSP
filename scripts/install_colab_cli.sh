#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

echo "=== 1. Check / Install uv ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

echo "=== 2. Install google-colab-cli with keyrings.alt ==="
uv tool install --force google-colab-cli --with keyrings.alt

echo "=== 3. Configure WSL2 keyring backend (PlaintextKeyring for headless WSL) ==="
mkdir -p "$HOME/.config/python_keyring"
cat > "$HOME/.config/python_keyring/keyringrc.cfg" << 'EOF'
[backend]
default-keyring=keyrings.alt.file.PlaintextKeyring
EOF
chmod 600 "$HOME/.config/python_keyring/keyringrc.cfg"

echo "=== 4. Verify colab CLI and keyring ==="
colab version
"$HOME/.local/share/uv/tools/google-colab-cli/bin/python" -c "import keyring; kr = keyring.get_keyring(); print('Active keyring:', kr)"
