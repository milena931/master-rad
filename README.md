# Master rad — Paralelizacija RL treniranja sa Ray

Projekat za master rad: **paralelizacija treniranja agentskih modela** u Gymnasium okruženjima sa **Ray RLlib**, lokalno i na **Google Cloud**.

Ista definicija eksperimenata (`config/experiments.yaml`) važi lokalno i na GCP. Klaster yaml-ovi u `gcp/` opisuju samo mašine.

## Šta projekat pokriva

| Komponenta | Opis |
|---|---|
| **Algoritmi** | PPO, APPO, DQN (diskretne akcije), SAC (kontinualne akcije) |
| **Okruženja** | CartPole-v1, LunarLander-v3, BipedalWalker-v3 |
| **Skalabilnost** | lokalno 1/2/4; GCP do 16 (CartPole/LL) odnosno 32 (BipedalWalker) |
| **Metrike** | Throughput (koraci/s), speedup, efikasnost, eval mean ± std |
| **GIF-ovi** | Random agent, naučeni agent, evolucija tokom treninga |
| **GCP** | Jedna VM za lake envove, multi-VM klaster za BipedalWalker |

Koji algoritam ide na koji env:

| | PPO | APPO | DQN | SAC |
|---|---|---|---|---|
| CartPole | da | da | da | — |
| LunarLander | da | da | da | — |
| BipedalWalker | da | da | — | da |

## Struktura

```
master-rad/
├── config/
│   └── experiments.yaml              # jedina definicija eksperimenata
├── src/
│   ├── run_experiments.py            # orkestrator (env × algo × workers)
│   ├── train_ray.py                  # PPO
│   ├── train_appo.py                 # APPO
│   ├── train_dqn.py                  # DQN
│   ├── train_sac.py                  # SAC
│   ├── evaluate_agent.py             # eval mean ± std + best GIF
│   ├── play_game.py                  # snimanje epizoda / evolution GIF
│   ├── plot_results.py               # grafikoni iz JSON rezultata
│   └── metrics.py                    # TrainingRun, JSON export
├── gcp/
│   ├── ray_cluster_lunarlander.yaml  # CartPole + LunarLander (1 VM)
│   ├── ray_cluster_bipedalwalker.yaml
│   ├── ray_cluster.yaml              # opšti multi-node (retko)
│   ├── setup_vm.sh
│   └── README.md
├── scripts/setup.sh
├── requirements.txt
└── results/                          # gitignored
    ├── *.json
    ├── checkpoints/
    ├── plots/
    └── gifs/
```

## Instalacija

```bash
cd master-rad
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# LunarLander i BipedalWalker (Box2D):
sudo apt-get install -y swig
pip install "gymnasium[box2d]"
```

## Pokretanje

Hiperparametri i broj iteracija su u `config/experiments.yaml`. Default workeri: `1 2 4`.

```bash
source .venv/bin/activate

# CartPole, svi diskretni algoritmi, GIF + eval
python src/run_experiments.py --envs cartpole \
    --algo ppo appo dqn --workers 4 --gif --evaluate 10

# LunarLander
python src/run_experiments.py --envs lunarlander \
    --algo ppo appo dqn --workers 4 --gif --evaluate 10

# BipedalWalker (lokalno w=4, ~30–60 min za PPO)
python src/run_experiments.py --envs bipedalwalker \
    --algo ppo appo sac --workers 4 --gif --evaluate 10
```

Samo PPO (default `--algo ppo`):

```bash
python src/run_experiments.py --envs cartpole --gif --evaluate 10
```

Jedan trening bez orkestratora:

```bash
python src/train_ray.py --env CartPole-v1 --workers 2 --iterations 20
```

Grafikoni iz `results/*.json`:

```bash
python src/plot_results.py
# PNG-ovi: results/plots/
```

## Okruženja

| Ključ | Env ID | Akcije | Budžet (iz yaml) | Gde |
|---|---|---|---|---|
| `cartpole` | CartPole-v1 | diskretne | PPO/APPO ~80 iter; DQN 100k koraka | lokalno ili GCP |
| `lunarlander` | LunarLander-v3 | diskretne | PPO/APPO ~1M koraka (65 × 16384); DQN 100k | lokalno ili GCP |
| `bipedalwalker` | BipedalWalker-v3 | kontinualne | PPO/APPO ~5.2M koraka (80 × 65536); SAC 512k | preporučeno GCP |

Ciljne nagrade: CartPole 450, LunarLander 200, BipedalWalker 300.

Tokom DQN treninga kolona **Reward** je ε-greedy. Greedy politiku gledaj u **Evolution** / `--evaluate`.

## Google Cloud Platform

Detaljnije: [`gcp/README.md`](gcp/README.md). Project: `master-rad-501412`, zona: `us-central1-a`.

Lake igre (brz env korak) idu na **jednu** VM — mreža između mašina usporava. BipedalWalker ide na **više** VM-ova jer je fizika spora.

| Eksperiment | Klaster | VM | Tip | vCPU | RAM | Cena/h |
|---|---|---|---|---|---|---|
| CartPole, LunarLander | `gcp/ray_cluster_lunarlander.yaml` | 1× head | **c2d-highcpu-16** | 16 | 32 GB | ~$0.60 |
| BipedalWalker | `gcp/ray_cluster_bipedalwalker.yaml` | head | **c3-standard-8** | 8 | 32 GB | ~$0.40 |
| BipedalWalker | isto | workeri 0–8, preemptible | **c3-highcpu-8** | 8 | 16 GB | ~$0.09 / VM |

```bash
# CartPole + LunarLander
ray up gcp/ray_cluster_lunarlander.yaml
ray attach gcp/ray_cluster_lunarlander.yaml
python src/run_experiments.py --envs cartpole lunarlander \
    --algo ppo appo dqn --workers 1 2 4 8 16 --gif --evaluate 10
ray down gcp/ray_cluster_lunarlander.yaml

# BipedalWalker
ray up gcp/ray_cluster_bipedalwalker.yaml
ray attach gcp/ray_cluster_bipedalwalker.yaml
python src/run_experiments.py --envs bipedalwalker \
    --algo ppo appo sac --workers 1 2 4 8 16 32 --gif --evaluate 10
ray rsync-down gcp/ray_cluster_bipedalwalker.yaml ~/master-rad/results/ ./results/gcp/
ray down gcp/ray_cluster_bipedalwalker.yaml
```

Na GCP: CartPole/LunarLander `--workers 1 2 4 8 16`; BipedalWalker isto plus **32**. **Obavezno** `ray down` kad završiš.

## Metrike

Iz `results/*.json` i `results/plots/`:

- **Throughput** (koraci/s) — koliko brže se skuplja iskustvo sa više workera
- **Speedup** — `T(w=1) / T(w=N)`
- **Efikasnost** — `speedup / N`
- **Eval mean ± std** — posle treninga, `--evaluate N`
- **Kriva učenja** — nagrada po iteraciji
