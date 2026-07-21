#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$HOME/GARVIS"
export GARVIS_ENABLE_RESEARCH=1
MISSION="$(python -c 'from garvis.economics.dream_cycle import DREAM_MISSION; print(DREAM_MISSION)')"
exec uv run --no-dev garvis --session procity-revenue-research "$MISSION"
