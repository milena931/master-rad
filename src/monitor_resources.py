"""
Snima CPU i memoriju mašine u CSV dok traje trening.

Koristi /proc (nema dodatnih paketa). Namenjeno laptopu i GCP VM-u.

Samostalno (druga terminal sesija, ili u pozadini):
  python src/monitor_resources.py --out results/monitor.csv --interval 2
  # Ctrl+C kad trening završi — ispisuje prosek/max

Uz orkestrator:
  python src/run_experiments.py --envs cartpole --algo ppo --monitor --gif

CSV kolone:
  timestamp, elapsed_sec, cpu_pct, load1, load5, nproc,
  mem_used_gb, mem_total_gb, mem_pct, ray_procs
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path


def _read_cpu_times() -> tuple[int, int]:
    """(idle+iowait, total) iz /proc/stat."""
    with open("/proc/stat", encoding="utf-8") as f:
        parts = f.readline().split()
    # cpu user nice system idle iowait irq softirq steal ...
    nums = [int(x) for x in parts[1:]]
    idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
    total = sum(nums)
    return idle, total


def _read_mem() -> tuple[float, float]:
    """(used_gb, total_gb) iz /proc/meminfo."""
    info: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            key, raw, *_ = line.split()
            info[key.rstrip(":")] = int(raw)
    total_kb = info["MemTotal"]
    avail_kb = info.get("MemAvailable", info.get("MemFree", 0))
    used_kb = total_kb - avail_kb
    return used_kb / (1024 * 1024), total_kb / (1024 * 1024)


def _count_ray_procs() -> int:
    count = 0
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/comm", encoding="utf-8") as f:
                    comm = f.read().strip().lower()
            except OSError:
                continue
            if "ray" in comm or comm in ("raylet", "gcs_server", "plasma_store"):
                count += 1
    except OSError:
        pass
    return count


def _cpu_percent(prev: tuple[int, int], cur: tuple[int, int]) -> float:
    d_idle = cur[0] - prev[0]
    d_total = cur[1] - prev[1]
    if d_total <= 0:
        return 0.0
    return 100.0 * (1.0 - d_idle / d_total)


def run_monitor(out_path: Path, interval: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop = {"flag": False}

    def _stop(*_args) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    prev = _read_cpu_times()
    time.sleep(min(interval, 1.0))
    t0 = time.time()
    rows: list[dict] = []

    fieldnames = [
        "timestamp",
        "elapsed_sec",
        "cpu_pct",
        "load1",
        "load5",
        "nproc",
        "mem_used_gb",
        "mem_total_gb",
        "mem_pct",
        "ray_procs",
    ]

    print(f"  [monitor] CPU/RAM → {out_path}  (interval {interval}s)", flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        while not stop["flag"]:
            cur = _read_cpu_times()
            cpu = _cpu_percent(prev, cur)
            prev = cur
            used_gb, total_gb = _read_mem()
            load1, load5, _load15 = os.getloadavg()
            now = time.time()
            row = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_sec": round(now - t0, 1),
                "cpu_pct": round(cpu, 1),
                "load1": round(load1, 2),
                "load5": round(load5, 2),
                "nproc": os.cpu_count() or 0,
                "mem_used_gb": round(used_gb, 3),
                "mem_total_gb": round(total_gb, 3),
                "mem_pct": round(100.0 * used_gb / total_gb, 1) if total_gb else 0.0,
                "ray_procs": _count_ray_procs(),
            }
            writer.writerow(row)
            f.flush()
            rows.append(row)
            deadline = now + interval
            while not stop["flag"] and time.time() < deadline:
                time.sleep(min(0.2, deadline - time.time()))

    _print_summary(out_path, rows)


def _print_summary(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        print("  [monitor] nema uzoraka", flush=True)
        return
    cpus = [r["cpu_pct"] for r in rows]
    mems = [r["mem_pct"] for r in rows]
    used = [r["mem_used_gb"] for r in rows]
    summary = out_path.with_suffix(".summary.txt")
    lines = [
        f"fajl:        {out_path}",
        f"uzoraka:     {len(rows)}",
        f"trajanje:    {rows[-1]['elapsed_sec']:.0f}s",
        f"CPU mean:    {sum(cpus)/len(cpus):.1f}%",
        f"CPU max:     {max(cpus):.1f}%",
        f"RAM mean:    {sum(mems)/len(mems):.1f}%  ({sum(used)/len(used):.2f} GB)",
        f"RAM max:     {max(mems):.1f}%  ({max(used):.2f} GB)",
        f"vCPU:        {rows[0]['nproc']}",
    ]
    text = "\n".join(lines) + "\n"
    summary.write_text(text, encoding="utf-8")
    print("\n  [monitor] sažetak:", flush=True)
    print(text, end="", flush=True)
    print(f"  [monitor] sažetak sačuvan: {summary}", flush=True)


def plot_csv(csv_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(df["elapsed_sec"], df["cpu_pct"], color="tab:blue")
    axes[0].set_ylabel("CPU (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["elapsed_sec"], df["mem_pct"], color="tab:orange")
    axes[1].set_ylabel("RAM (%)")
    axes[1].set_xlabel("vreme (s)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(csv_path.name)
    fig.tight_layout()
    png = csv_path.with_suffix(".png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"  [monitor] grafikon: {png}")
    return png


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snimi CPU i RAM u CSV dok traje proces",
    )
    parser.add_argument("--out", default=None, help="Putanja do CSV fajla (snimanje)")
    parser.add_argument("--interval", type=float, default=2.0, help="Sekundi između uzoraka")
    parser.add_argument(
        "--plot",
        default=None,
        metavar="CSV",
        help="Nacrtaj CPU/RAM grafikon iz postojećeg CSV-a (ne snima)",
    )
    args = parser.parse_args()
    if args.plot:
        plot_csv(Path(args.plot))
        return
    if not args.out:
        parser.error("treba --out PATH ili --plot CSV")
    if not Path("/proc/stat").exists():
        print("  [monitor] treba Linux /proc — ovaj skript nije za ovaj OS.", file=sys.stderr)
        sys.exit(1)
    run_monitor(Path(args.out), max(args.interval, 0.5))


if __name__ == "__main__":
    main()
