#!/usr/bin/env bash
# =============================================================================
# setup_vm.sh — Podizanje GCP VM za master rad PoC
#
# Pokretanje na svežem GCP Ubuntu 22.04 VM:
#   chmod +x setup_vm.sh && ./setup_vm.sh
#
# Preporučena VM konfiguracija (unutar $300 studentskog kredita):
#   Machine type: e2-standard-8  (8 vCPU, 32 GB RAM) ~ $0.27/h
#   ili za GPU:   n1-standard-4 + 1x T4 GPU          ~ $0.35/h
#   Region: us-central1 (jeftinije)
#   OS: Ubuntu 22.04 LTS
#
# Procena troškova za eksperimente:
#   CartPole/LunarLander:  ~2-4h  = $0.54-$1.08  (bez GPU)
#   Atari Pong (sa GPU):   ~4-8h  = $1.40-$2.80
#   Ukupno za master rad:  <$20   ostaje dosta kredita
# =============================================================================
set -euo pipefail

echo "=== [1/5] System update ==="
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv python3-dev git htop

echo ""
echo "=== [2/5] Python venv ==="
python3 -m venv ~/ray-env
source ~/ray-env/bin/activate

echo ""
echo "=== [3/5] Instalacija Python zavisnosti ==="
pip install --upgrade pip
pip install "ray[rllib]>=2.40.0" \
            "gymnasium[classic-control,box2d]>=1.0.0" \
            "torch>=2.0.0" \
            "matplotlib>=3.8.0" \
            "pyyaml>=6.0" \
            "imageio>=2.33.0"

echo ""
echo "=== [4/5] Kloniranje projekta ==="
# Ako projekat imaš na GitHub-u:
#   git clone https://github.com/TVOJ_USERNAME/master-rad.git ~/master-rad
# Ili kopiraj fajlove ručno (scp):
#   scp -r ./rad USERNAME@VM_IP:~/master-rad
mkdir -p ~/master-rad
echo "  → Kopiraj projekat u ~/master-rad (git clone ili scp)"

echo ""
echo "=== [5/5] Pokretanje Ray head nodea ==="
# Podiže Ray na ovoj mašini (lokalni klaster sa svim CPU jezgrima)
ray start --head \
          --dashboard-host=0.0.0.0 \
          --dashboard-port=8265 \
          --num-cpus=$(nproc) \
          --block &

echo ""
echo "=========================================="
echo "  Setup završen!"
echo ""
echo "  Ray dashboard: http://$(curl -s ifconfig.me):8265"
echo ""
echo "  Pokreni eksperiment:"
echo "    source ~/ray-env/bin/activate"
echo "    cd ~/master-rad/src"
echo "    python run_experiments.py --envs lunarlander"
echo ""
echo "  Ili sa --ray-address (ako koristiš više VM-ova):"
echo "    python train_ray.py --env LunarLander-v3 --workers 8 \\"
echo "      --ray-address 'ray://localhost:6379'"
echo "=========================================="
