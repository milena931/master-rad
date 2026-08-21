# Google Cloud Platform — Vodič za master rad

Google Cloud studentski kredit: **$300** za nove naloge (90 dana). Za ove eksperimente treba manje od **$30**.

Hiperparametri: samo `config/experiments.yaml` (isto kao lokalno).
Klaster yaml-ovi ispod podižu mašine, ne menjaju trening.

---

## Mašine koje koristiš

Project: `master-rad-501412`, zona: `us-central1-a`.

| Eksperiment | Fajl | VM | Tip | vCPU / RAM | Cena/h |
|---|---|---|---|---|---|
| CartPole, LunarLander | `ray_cluster_lunarlander.yaml` | 1× head | **c2d-highcpu-16** | 16 / 32 GB | ~$0.60 |
| BipedalWalker | `ray_cluster_bipedalwalker.yaml` | head | **c3-standard-8** | 8 / 32 GB | ~$0.40 |
| BipedalWalker | isto | 0–8 workers, preemptible | **c3-highcpu-8** | 8 / 16 GB | ~$0.09 / VM |

CartPole/LunarLander su na **jednoj** VM jer je env korak brz — mreža između VM-ova usporava.
BipedalWalker je na **više** VM-ova jer je fizika spora, pa se distribuiranje isplati.

```bash
# CartPole + LunarLander
ray up gcp/ray_cluster_lunarlander.yaml
ray attach gcp/ray_cluster_lunarlander.yaml
# Na VM: cd ~/master-rad && python src/run_experiments.py ... | tee results/run.log
# Sa laptopa PRE ray down:
mkdir -p results/gcp
ray rsync-down gcp/ray_cluster_lunarlander.yaml ~/master-rad/results/ ./results/gcp/
ray down gcp/ray_cluster_lunarlander.yaml

# BipedalWalker
ray up gcp/ray_cluster_bipedalwalker.yaml
ray attach gcp/ray_cluster_bipedalwalker.yaml
mkdir -p results/gcp
ray rsync-down gcp/ray_cluster_bipedalwalker.yaml ~/master-rad/results/ ./results/gcp/
ray down gcp/ray_cluster_bipedalwalker.yaml
```

---

## Opcija A — Jedna VM ručno (alternativa, bez Ray autoscalera)

### Korak 1 — Napravi VM na GCP

```bash
# Instaliraj gcloud CLI (jednom)
curl https://sdk.cloud.google.com | bash
gcloud init

# Napravi VM (16 jezgara, 32 GB RAM — isto kao ray_cluster_lunarlander.yaml)
gcloud compute instances create master-rad-vm \
  --machine-type=c2d-highcpu-16 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=60GB \
  --zone=us-central1-a

# Poveži se SSH
gcloud compute ssh master-rad-vm --zone=us-central1-a
```

**Cena:** c2d-highcpu-16 = **~$0.60/h** → za 2h eksperimenata ≈ $1.20

### Korak 2 — Setup na VM

```bash
# Na VM (posle SSH):
curl -O https://raw.githubusercontent.com/.../setup_vm.sh
chmod +x setup_vm.sh && ./setup_vm.sh
```

Ili ručno:
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv ~/ray-env && source ~/ray-env/bin/activate
pip install "ray[rllib]" "gymnasium[classic-control,box2d]" torch matplotlib pyyaml
```

### Korak 3 — Kopiraj projekat na VM

```bash
# Sa LOKALNOG računara:
gcloud compute scp --recurse /home/milena/Documents/etf/master/rad \
  master-rad-vm:~/master-rad --zone=us-central1-a
```

### Korak 4 — Pokreni eksperiment

```bash
# Na VM:
source ~/ray-env/bin/activate
cd ~/master-rad

# Lokalni Ray klaster (sva jezgra na ovoj VM)
ray start --head --num-cpus=16

python src/run_experiments.py --envs cartpole lunarlander \
    --algo ppo appo dqn --workers 1 2 4 8 16 --gif --evaluate 10 --monitor
```

### Korak 5 — Preuzmi rezultate

```bash
# Sa lokalnog računara:
gcloud compute scp --recurse master-rad-vm:~/master-rad/results \
  /home/milena/Documents/etf/master/rad/results --zone=us-central1-a
```

### VAŽNO — Ugasi VM kad završiš!

```bash
gcloud compute instances stop master-rad-vm --zone=us-central1-a
# ili je obriši:
gcloud compute instances delete master-rad-vm --zone=us-central1-a
```

---

## Opcija B — Multi-node (BipedalWalker)

Koristi `gcp/ray_cluster_bipedalwalker.yaml` (c3-standard-8 head + c3-highcpu-8 preemptible workeri).
Ne koristi stari `ray_cluster.yaml` osim ako namerno hoćeš manji head (c3-standard-4).

```bash
pip install "ray[default]"
ray up gcp/ray_cluster_bipedalwalker.yaml
ray attach gcp/ray_cluster_bipedalwalker.yaml
python src/run_experiments.py --envs bipedalwalker \
    --algo ppo appo sac --workers 1 2 4 8 16 32 --gif --evaluate 10 --monitor
ray down gcp/ray_cluster_bipedalwalker.yaml
```

---

## Preporučeni redosled za master rad

| Faza | Gde | Šta |
|---|---|---|
| 1. Razvoj | Lokalno | CartPole, debug, mali run-ovi |
| 2. CartPole + LunarLander | GCP c2d-highcpu-16 | PPO / APPO / DQN, w=1,2,4,8,16 |
| 3. BipedalWalker | GCP c3-standard-8 head + c3-highcpu-8 workeri | PPO / APPO / SAC, w=1,2,4,8,16,32 |

---

## Korisni gcloud komande

```bash
gcloud compute instances list              # lista VM-ova
gcloud compute instances start VM_NAME     # pokretanje
gcloud compute instances stop VM_NAME      # zaustavljanje (prestaje naplata)
gcloud billing accounts list               # pregled kredita
```
