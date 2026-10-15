#!/usr/bin/env python3
"""Record the peak memory and wall time of every VIBE build process.

VIBE stores build_time and index_size (memory kept after the build) per
build, but not peak memory. Each build runs in its own process
(run_algorithm.py: load data, build, run every search setting), so this
polls /proc for those processes and logs each one's peak resident memory,
the kernel's high-water mark (VmHWM), when it exits. Only a peak in the last
interval before exit can be missed.

    python tools/vibe_memwatch.py watch OUT.csv [--interval 1]   # until Ctrl-C / SIGTERM
    python tools/vibe_memwatch.py summary OUT.csv [--limit-gb 16] [--timeout-h 24]

Start `watch` before run.py and stop it afterwards; rows are appended, so
one file can cover several runs.
"""
import argparse
import csv
import os
import signal
import sys
import time

FIELDS = ["dataset", "algorithm", "args", "pid", "start", "end", "wall_s", "peak_rss_gb"]


def _builds():
    """{pid: (dataset, algorithm, args)} for the python processes running run_algorithm.py."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        # numactl and the Apptainer starter carry the same arguments; keep the interpreter
        if "run_algorithm.py" not in argv or not os.path.basename(argv[0]).startswith("python"):
            continue
        get = lambda flag: argv[argv.index(flag) + 1] if flag in argv else ""
        i = argv.index("run_algorithm.py")
        build_args = next((a for a in argv[i:] if a.startswith("[")), "")
        out[int(pid)] = (get("--dataset"), get("--algorithm"), build_args)
    return out


def _hwm_kb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def watch(path, interval):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, FIELDS)
    if new:
        w.writeheader()
        f.flush()
    live = {}  # pid -> [dataset, algorithm, args, start, peak_kb]
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    def emit(pid, end):
        d, a, args, start, peak = live.pop(pid)
        w.writerow(dict(dataset=d, algorithm=a, args=args, pid=pid,
                        start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)),
                        end=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end)),
                        wall_s="%.0f" % (end - start), peak_rss_gb="%.3f" % (peak / 2**20)))
        f.flush()

    print(f"watching VIBE build processes every {interval}s -> {path}", file=sys.stderr)
    try:
        while not stop:
            now = time.time()
            cur = _builds()
            for pid, meta in cur.items():
                if pid not in live:
                    live[pid] = [*meta, now, 0]
                hwm = _hwm_kb(pid)
                if hwm is not None:
                    live[pid][4] = max(live[pid][4], hwm)
            for pid in [p for p in live if p not in cur]:
                emit(pid, now)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    for pid in list(live):  # still running when stopped: log what was seen so far
        emit(pid, time.time())
    f.close()


def summary(path, limit_gb, timeout_h):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["algorithm"]), []).append(r)
    print("%-34s %-18s %6s %10s %10s %8s %10s" % (
        "dataset", "algorithm", "builds", "peak GB", "median GB", ">%g GB" % limit_gb, "max wall h"))
    for (d, a), rs in sorted(groups.items()):
        peaks = sorted(float(r["peak_rss_gb"]) for r in rs)
        walls = [float(r["wall_s"]) / 3600 for r in rs]
        print("%-34s %-18s %6d %10.1f %10.1f %8d %10.2f%s" % (
            d, a, len(rs), peaks[-1], peaks[len(peaks) // 2], sum(p > limit_gb for p in peaks), max(walls),
            "  <-- over the %g h timeout" % timeout_h if max(walls) > timeout_h else ""))
    worst = max(rows, key=lambda r: float(r["peak_rss_gb"]), default=None)
    if worst:
        print("\nlargest peak: %s GB, %s %s %s" % (worst["peak_rss_gb"], worst["dataset"], worst["algorithm"], worst["args"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("watch")
    p.add_argument("out")
    p.add_argument("--interval", type=float, default=1.0)
    p = sub.add_parser("summary")
    p.add_argument("out")
    p.add_argument("--limit-gb", type=float, default=16.0, help="VIBE's README recommends 16 GB per core")
    p.add_argument("--timeout-h", type=float, default=24.0, help="run.py --timeout default")
    a = ap.parse_args()
    if a.cmd == "watch":
        watch(a.out, a.interval)
    else:
        summary(a.out, a.limit_gb, a.timeout_h)


if __name__ == "__main__":
    main()
