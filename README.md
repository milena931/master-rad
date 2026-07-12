# Master rad PoC — Paralelizacija RL treniranja sa Ray

Proof-of-concept za master rad: **paralelizacija treniranja agentskih modela u igračkim okruženjima korišćenjem Ray RLlib biblioteke i Google Cloud Platform resursa**.

## Šta PoC pokriva

| Komponenta | Opis |
|---|---|
| **Ray RLlib** | Distribuirano PPO treniranje sa N paralelnih `env_runner` procesa |
| **Gymnasium** | CartPole-v1, LunarLander-v3, BipedalWalker-v3 |
| **Metrike** | Throughput (koraci/sec), speedup, efikasnost, JSON logovi |
| **GIF vizualizacija** | Random agent, naučeni agent, evolucija tokom treninga |
| **GCP klaster** | Ray multi-node klaster na Google Cloud VM-ovima |

## Struktura projekta

```
rad/
├── config/
│   └── experiments.yaml      # Env-i, PPO hiperparametri, worker counts
├── src/
│   ├── train_ray.py          # Distribuirano treniranje (Ray RLlib)
│   ├── run_experiments.py    # Orchestracija skalabilnost eksperimenata
│   ├── plot_results.py       # Generisanje grafikona iz JSON rezultata
│   ├── play_game.py          # Snimanje GIF-ova (random, naučen, evolucija)
│   └── metrics.py            # TrainingRun dataclass, JSON export
├── gcp/
│   ├── ray_cluster.yaml      # Ray autoscaler konfiguracija za GCP
│   └── setup_vm.sh           # Setup skripta za GCP Ubuntu VM
├── scripts/
│   └── setup.sh              # Lokalni setup
├── requirements.txt
└── results/                  # Output (gitignored)
    ├── *.json                # Metrike po run-u
    ├── plots/                # PNG grafikoni
    └── gifs/                 # Animacije agenta
```

## Instalacija

```bash
# Klonirati repozitorijum i ući u folder
cd rad

# Kreirati virtuelno okruženje i instalirati zavisnosti
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Za LunarLander i BipedalWalker potreban je sistem-level swig:
sudo apt-get install -y swig
pip install "gymnasium[box2d]"
```

## Pokretanje eksperimenata

### Brzi test (CartPole, ~2 min)

```bash
cd /putanja/do/rad
source .venv/bin/activate

python src/train_ray.py \
    --env CartPole-v1 \
    --workers 2 \
    --iterations 20
```

### Skalabilnost eksperiment — sva tri okruženja

```bash
# CartPole i LunarLander (lokalno, ~30 min)
python src/run_experiments.py \
    --envs cartpole lunarlander \
    --gif

# BipedalWalker — scaling study (merenje throughputa, ~80 min)
python src/run_experiments.py \
    --envs bipedalwalker \
    --workers 4 \
    --scaling-only
```

### BipedalWalker — puno treniranje (lokalno, ~2h sa 4 workera)

```bash
python src/run_experiments.py \
    --envs bipedalwalker \
    --workers 4 \
    --gif
```

### Grafikoni

```bash
python src/plot_results.py
# PNG-ovi se čuvaju u results/plots/
```

## Okruženja

| Ključ | Env ID | Zahtevnost | Preporučeno |
|---|---|---|---|
| `cartpole` | CartPole-v1 | Lako | Lokalno, ~2 min |
| `lunarlander` | LunarLander-v3 | Srednje | Lokalno, ~25 min (w=4) |
| `bipedalwalker` | BipedalWalker-v3 | Teško | GCP, ~30 min (w=8) |

## Google Cloud Platform

### Pokretanje Ray klastera na GCP

```bash
# 1. Instalirati gcloud CLI i autentifikovati se
gcloud auth login
gcloud auth application-default login
gcloud config set project TVOJ_PROJECT_ID

# 2. Uključiti potrebne API-je
gcloud services enable compute.googleapis.com iam.googleapis.com cloudresourcemanager.googleapis.com

# 3. Pokrenuti klaster (kreira VM-ove automatski)
ray up gcp/ray_cluster.yaml

# 4. SSH na head node i pokrenuti trening
ray attach gcp/ray_cluster.yaml
# Na VM-u:
cd ~/master-rad
python src/run_experiments.py \
    --envs bipedalwalker \
    --workers 8 \
    --gif

# 5. Kopirati rezultate lokalno
ray rsync-down gcp/ray_cluster.yaml ~/master-rad/results/ ./results/gcp/

# 6. Ugasiti klaster (OBAVEZNO — zaustavlja naplatu)
ray down gcp/ray_cluster.yaml
```

### Preporučena GCP konfiguracija

| VM | Tip | vCPU | RAM | Cena/h |
|---|---|---|---|---|
| Head node | e2-standard-4 | 4 | 16 GB | ~$0.13 |
| Worker node | e2-standard-8 | 8 | 32 GB | ~$0.27 |
| **Ukupno** | | **12** | **48 GB** | **~$0.40** |

Ceo BipedalWalker eksperiment: ~30 min = **~$0.20**

## Metrike za master rad

Iz `results/*.json` i grafikona izvlačiš:

- **Throughput** (koraci/sec): koliko brže skuplja iskustvo sa više workera
- **Speedup**: `T(w=1) / T(w=N)` — koliko puta je brže sa N workera
- **Efikasnost**: `speedup / N` — koliko dobro koristimo dodatne resurse
- **Kriva učenja**: nagrada tokom iteracija po worker konfiguraciji

## Ray RLlib skalabilnost

| Workers | Speedup | Efikasnost |
|---|---|---|
| 1 | 1.00x | 1.00 |
| 2 | ~1.8x | ~0.90 |
| 4 | ~3.1x | ~0.78 |
