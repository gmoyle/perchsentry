import time
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime

log = logging.getLogger("birdbuddy")

# Kernel thermal zone — millidegrees C, no subprocess needed for the reading.
_THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")

TEMP_LOG = Path(__file__).parent / "logs" / "temps.csv"
SAMPLE_SECS = 60           # how often to record a point
RETENTION_HOURS = 48       # how much history to keep on disk
_PRUNE_EVERY = 60          # rewrite/prune the file every N samples


def read_temp():
    """Current SoC temperature plus decoded throttle state.

    The firmware soft-throttles the CPU at ~80°C, hard-throttles at ~85°C, and
    shuts down at 110°C, so this is informational — but it lets the dashboard
    flag heat trouble before it costs frames."""
    temp_c = None
    try:
        temp_c = round(int(_THERMAL_ZONE.read_text().strip()) / 1000, 1)
    except Exception:
        pass

    # `vcgencmd get_throttled` returns a bit field. Low bits = happening now,
    # high bits (>>16) = has happened since boot.
    throttled_now = capped_now = undervolt_now = False
    throttled_ever = undervolt_ever = False
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3
        ).stdout.strip()
        flags = int(out.split("=")[1], 16)
        undervolt_now = bool(flags & 0x1)
        capped_now = bool(flags & 0x2)
        throttled_now = bool(flags & 0x4)
        undervolt_ever = bool(flags & 0x10000)
        throttled_ever = bool(flags & 0x40000)
    except Exception:
        pass

    if temp_c is None:
        status = "unknown"
    elif throttled_now or temp_c >= 82:
        status = "danger"
    elif capped_now or throttled_ever or undervolt_now or temp_c >= 75:
        status = "warn"
    else:
        status = "ok"

    return {
        "temp_c": temp_c,
        "status": status,
        "throttled_now": throttled_now,
        "capped_now": capped_now,
        "undervolt_now": undervolt_now,
        "throttled_ever": throttled_ever,
        "undervolt_ever": undervolt_ever,
    }


def _prune(cutoff_epoch):
    """Rewrite the log keeping only rows newer than cutoff_epoch."""
    try:
        lines = TEMP_LOG.read_text().splitlines()
    except FileNotFoundError:
        return
    kept = []
    for ln in lines:
        try:
            if float(ln.split(",")[0]) >= cutoff_epoch:
                kept.append(ln)
        except (ValueError, IndexError):
            continue
    TEMP_LOG.write_text("\n".join(kept) + ("\n" if kept else ""))


def temp_history(hours=24, max_points=240):
    """Return [{"t": epoch_secs, "temp": c}] within the window, bucket-averaged
    down to at most max_points so the chart stays light regardless of range."""
    cutoff = time.time() - hours * 3600
    rows = []
    try:
        for ln in TEMP_LOG.read_text().splitlines():
            try:
                ep, temp = ln.split(",")[:2]
                ep = float(ep)
                if ep >= cutoff and temp:
                    rows.append((ep, float(temp)))
            except (ValueError, IndexError):
                continue
    except FileNotFoundError:
        return []

    if len(rows) <= max_points:
        return [{"t": int(ep), "temp": round(t, 1)} for ep, t in rows]

    # Bucket by time so gaps don't distort the shape.
    span = rows[-1][0] - rows[0][0]
    bucket = span / max_points if span > 0 else 1
    out, cur_key, acc = [], None, []
    for ep, t in rows:
        key = int((ep - rows[0][0]) // bucket)
        if cur_key is None:
            cur_key = key
        if key != cur_key and acc:
            mid = sum(e for e, _ in acc) / len(acc)
            avg = sum(v for _, v in acc) / len(acc)
            out.append({"t": int(mid), "temp": round(avg, 1)})
            acc, cur_key = [], key
        acc.append((ep, t))
    if acc:
        mid = sum(e for e, _ in acc) / len(acc)
        avg = sum(v for _, v in acc) / len(acc)
        out.append({"t": int(mid), "temp": round(avg, 1)})
    return out


class TempLogger:
    """Samples the SoC temperature on an interval and appends it to a bounded
    CSV (epoch,temp_c), pruning to RETENTION_HOURS periodically."""

    def __init__(self):
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        TEMP_LOG.parent.mkdir(exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        n = 0
        while not self._stop.is_set():
            t = read_temp().get("temp_c")
            if t is not None:
                try:
                    with open(TEMP_LOG, "a") as f:
                        f.write(f"{int(time.time())},{t}\n")
                except Exception as e:
                    log.warning(f"Temp log write failed: {e}")
            n += 1
            if n % _PRUNE_EVERY == 0:
                _prune(time.time() - RETENTION_HOURS * 3600)
            self._stop.wait(SAMPLE_SECS)
