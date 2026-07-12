# Google Cloud Platform — Vodič za master rad

## Studentski kredit

Google Cloud daje **$300 besplatnih kredita** za nove naloge (90 dana).

Registracija: https://cloud.google.com/free  
(treba ti kredit kartica ali se ne naplaćuje dok si u free trial-u)

Za master rad PoC ti treba manje od **$30** od tih $300.

---

## Opcija A — Jedna VM (preporučeno, najjednostavnije)

### Korak 1 — Napravi VM na GCP

```bash
# Instaliraj gcloud CLI (jednom)
curl https://sdk.cloud.google.com | bash
gcloud init

# Napravi VM (8 jezgara, 32 GB RAM, Ubuntu 22.04)
gcloud compute instances create master-rad-vm \
  --machine-type=e2-standard-8 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=60GB \
  --zone=us-central1-a

# Poveži se SSH
gcloud compute ssh master-rad-vm --zone=us-central1-a
```

**Cena:** e2-standard-8 = **~$0.27/h** → za 4h eksperimenata ≈ $1.08

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
cd ~/master-rad/src

# Lokalni Ray klaster (koristi svih 8 jezgara)
ray start --head --num-cpus=8

# LunarLander skalabilnost: 1, 2, 4, 8 workera
python run_experiments.py --envs lunarlander
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

## Opcija B — Multi-node klaster (napredna, nije obavezna)

Za pravi distribuirani Ray klaster na više VM-ova, koristi `ray_cluster.yaml`:

```bash
# Na lokalnom računaru (gde imaš gcloud i ray instaliran):
pip install "ray[default]"

# Uređaj project_id u ray_cluster.yaml, pa:
ray up gcp/ray_cluster.yaml

# Pokreni eksperiment na klasteru:
ray submit gcp/ray_cluster.yaml src/train_ray.py \
  --env LunarLander-v3 --workers 8 --iterations 200

# OBAVEZNO ugasiti klaster kad završiš (inače naplaćuje!):
ray down gcp/ray_cluster.yaml
```

---

## Preporučeni redosled za master rad

| Faza | Gde | Šta |
|---|---|---|
| 1. Razvoj | Lokalno | CartPole, debug, mali run-ovi |
| 2. Eksperimenti | GCP VM (e2-standard-8) | LunarLander 1/2/4/8 workera |
| 3. Demo | GCP ili lokalno | Snimanje GIF agenta (Pong opciono) |

---

## Koliko košta šta

| VM tip | vCPU | RAM | Cena/h | Preporučeno za |
|---|---|---|---|---|
| e2-standard-4 | 4 | 16 GB | ~$0.13 | CartPole, testiranje |
| **e2-standard-8** | **8** | **32 GB** | **~$0.27** | **LunarLander (preporučeno)** |
| n1-standard-4 + T4 GPU | 4 | 15 GB | ~$0.35 | Atari Pong |

Za pun skalabilnost eksperiment (LunarLander, 4 worker konfiguracije):  
~4h na e2-standard-8 = **~$1.08** od $300 kredita.

---

## Korisni gcloud komande

```bash
gcloud compute instances list              # lista VM-ova
gcloud compute instances start VM_NAME     # pokretanje
gcloud compute instances stop VM_NAME      # zaustavljanje (prestaje naplata)
gcloud billing accounts list               # pregled kredita
```
