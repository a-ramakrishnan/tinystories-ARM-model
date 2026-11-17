# J3: run a long training job so that it can be watched, and restarted, without the browser tab.
#
#   python3 -m infra.colab_job --name f6-2p5b --dir "$OUT" --retries 3 -- \
#       python3 -u -m diffusion_models.train --mode f6-124m ... --resume
#
# The command runs as a child process. Its output is shown as usual and appended to <dir>/job.log, and every
# progress line it prints (the trainers' "update 12,000/152,600 ... tokens/s", pretrain's tqdm bar, "eval @ ..."
# and "Step N: ... val loss" lines) updates <dir>/status.json: step, total, loss, speed, ETA, the last evaluation,
# and when it was written. With --retries, a command that exits with an error is started again (the trainers'
# --resume picks up from the last saved state). A runtime that dies outright can't report anything, but its
# status.json stops being updated: `python3 -m infra.colab_job show <dir>` says "stale" after a few minutes.
#
# Colab keeps a runtime only while its browser tab is open (except with background execution on Colab Pro+), so
# this doesn't keep a job alive; it makes a dead one obvious and its restart one cell.
import argparse
import json
import os
import re
import subprocess
import sys
import time

STATUS, LOG = "status.json", "job.log"
STALE_AFTER = 300          # seconds without an update before `show` calls a running job stale
WRITE_EVERY = 30           # seconds between status writes while progress lines arrive

_num = lambda s: float(s.replace(",", ""))
PATTERNS = [
    # diffusion_models/train.py and sft.py: "update 12,000/152,600  loss 5.1234  lr 4.2e-04  142,334 tokens/s"
    ("progress", re.compile(r"\bupdate\s+([\d,]+)\s*/\s*([\d,]+).*?loss\s+([\d.]+)(?:.*?([\d,]+)\s+tokens/s)?")),
    # tqdm bars (pretrain.py, stage2.py): " 12%|#   | 18000/152600 [29:01<3:33:10, 10.52it/s]"
    ("tqdm", re.compile(r"\|\s*(\d+)\s*/\s*(\d+)\s*\[[^\]]*?([\d.]+)\s*it/s")),
    # evaluations: "eval @ 24,000: masked-token loss 5.0529 | likelihood bound 4.2297 ..." or "Step 2000: train loss 3.1, val loss 3.2"
    ("eval", re.compile(r"(eval @ [\d,]+:.*|Step \d+: .*val loss.*)")),
    ("resume", re.compile(r"(Resuming at (?:update|step) [\d,]+.*)")),
]


def parse(line, st):
    """Fold one output line into the status dict; True if it carried progress worth writing."""
    for kind, rx in PATTERNS:
        m = rx.search(line)
        if not m:
            continue
        if kind == "progress":
            st["step"], st["total"], st["loss"] = int(_num(m[1])), int(_num(m[2])), float(m[3])
            if m[4]:
                st["tokens_per_sec"] = int(_num(m[4]))
        elif kind == "tqdm":
            st["step"], st["total"], st["steps_per_sec"] = int(m[1]), int(m[2]), float(m[3])
        elif kind == "eval":
            st["last_eval"] = m[1].strip()[:300]
        elif kind == "resume":
            st["resumed"] = m[1].strip()[:300]
        return True
    return False


def eta(st, now):
    """Seconds left, from the step rate seen in this attempt (robust to resumes and compile warm-up)."""
    s0, t0 = st.get("_rate_from", (None, None))
    if st.get("step") is None or not st.get("total"):
        return None
    if s0 is None or st["step"] < s0:
        st["_rate_from"] = (st["step"], now)
        return None
    if st["step"] == s0 or now <= t0:
        return None
    rate = (st["step"] - s0) / (now - t0)
    st["steps_per_sec_measured"] = round(rate, 3)
    return (st["total"] - st["step"]) / rate if rate > 0 else None


def write_status(d, st):
    now = time.time()
    left = eta(st, now)
    out = {k: v for k, v in st.items() if not k.startswith("_")}
    out.update(updated_at=time.strftime("%Y-%m-%d %H:%M:%S %Z"), updated_ts=round(now),
               eta_seconds=round(left) if left else None,
               eta_at=time.strftime("%H:%M %Z", time.localtime(now + left)) if left else None)
    tmp = os.path.join(d, STATUS + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, os.path.join(d, STATUS))
    return out


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def run(cmd, d, name, retries=0, retry_wait=30, quiet_pattern=None, echo=True):
    """Run cmd with status reporting; returns the final exit code."""
    os.makedirs(d, exist_ok=True)
    st = {"job": name, "command": " ".join(cmd), "state": "starting", "attempt": 0, "gpu": gpu_name(),
          "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    quiet = re.compile(quiet_pattern, re.I) if quiet_pattern else None
    code = None
    with open(os.path.join(d, LOG), "a") as log:
        for attempt in range(1, retries + 2):
            st.update(attempt=attempt, state="running", exit_code=None)
            st.pop("_rate_from", None)
            log.write(f"\n=== {name} attempt {attempt} at {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
            write_status(d, st)
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            last_write, buf = 0.0, b""
            while True:
                chunk = p.stdout.read(4096)
                if not chunk:
                    if not buf:
                        break
                    chunk, buf = buf + b"\n", b""     # the last line, if it had no newline
                buf += chunk
                parts = re.split(rb"[\r\n]", buf)     # tqdm redraws with \r
                buf = parts.pop()
                changed = False
                for raw in parts:
                    line = raw.decode("utf-8", "replace").rstrip()
                    if not line:
                        continue
                    log.write(line + "\n")
                    if echo and not (quiet and quiet.search(line)):
                        print(line, flush=True)
                    changed |= parse(line, st)
                    st["last_line"] = line[:300]
                log.flush()
                if changed and time.time() - last_write > WRITE_EVERY:
                    write_status(d, st)
                    last_write = time.time()
            code = p.wait()
            st["exit_code"] = code
            if code == 0:
                st["state"] = "finished"
                break
            st["state"] = "failed" if attempt > retries else "restarting"
            write_status(d, st)
            if attempt > retries:
                break
            time.sleep(retry_wait)
        st["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        write_status(d, st)
    return code


def show(path, now=None):
    """A one-paragraph summary of a status.json (or a folder holding one)."""
    if os.path.isdir(path):
        path = os.path.join(path, STATUS)
    st = json.load(open(path))
    now = now or time.time()
    state = st.get("state")
    if state in ("running", "starting", "restarting") and now - st.get("updated_ts", now) > STALE_AFTER:
        state = f"stale (no update for {int((now - st['updated_ts']) / 60)} min: the runtime has probably stopped)"
    parts = [f"{st.get('job')}: {state}"]
    if st.get("step") is not None and st.get("total"):
        parts.append(f"step {st['step']:,}/{st['total']:,} ({st['step'] / st['total']:.0%})")
    if st.get("loss") is not None:
        parts.append(f"loss {st['loss']}")
    if st.get("tokens_per_sec"):
        parts.append(f"{st['tokens_per_sec']:,} tokens/s")
    if st.get("eta_at") and not state.startswith("stale") and state != "finished":
        parts.append(f"done about {st['eta_at']}")
    out = " · ".join(parts)
    if st.get("last_eval"):
        out += f"\n  last eval: {st['last_eval']}"
    out += f"\n  updated {st.get('updated_at')}, attempt {st.get('attempt')}" + (
        f", exit code {st['exit_code']}" if st.get("exit_code") not in (None, 0) else "")
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "show":
        for p in argv[1:]:
            print(show(p))
        return 0
    ap = argparse.ArgumentParser(description="Run a training command with a status.json heartbeat and retries.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True, help="folder for status.json and job.log (e.g. the run folder on Drive)")
    ap.add_argument("--retries", type=int, default=0, help="restart a failed command this many times")
    ap.add_argument("--retry-wait", type=int, default=30)
    ap.add_argument("--quiet", default="warning", help="regex of output lines not to echo (still logged)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("give the command after --")
    return run(cmd, a.dir, a.name, a.retries, a.retry_wait, a.quiet)


if __name__ == "__main__":
    sys.exit(main())
