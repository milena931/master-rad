#!/usr/bin/env bash
# Setup okruženja za master rad
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Kreiranje virtualnog okruženja ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Instalacija zavisnosti ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Instalacija Box2D za LunarLander (opciono) ==="
pip install "gymnasium[box2d]" 2>/dev/null || echo "Box2D nije instaliran — LunarLander neće raditi"

echo ""
echo "=== Setup završen ==="
echo "Aktiviraj okruženje: source .venv/bin/activate"
echo ""
echo "Brzi test:"
echo "  cd src && python train_ray.py --env CartPole-v1 --workers 2 --iterations 10"
echo ""
echo "Pun eksperiment skalabilnosti:"
echo "  cd src && python run_experiments.py --envs cartpole"
