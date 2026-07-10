"""
Nightly capture curator.

Real-time now captures liberally: any frame the yolov8 NPU pre-filter calls a
"bird" is kept, without gating on the (unreliable) SpeciesNet confidence. That
means the daytime capture folder collects some false positives — a swaying
chain, a shadow, the red feeder — that SpeciesNet will happily mislabel as a
hummingbird.

This pass runs after midnight and culls those, using two signals that are far
more trustworthy than SpeciesNet confidence and cost nothing to compute (both
are already in the log from capture time):

  1. Burst membership — a real visit produces a CLUSTER of captures over
     seconds; a spurious trigger is usually an isolated one-off.
  2. yolov8 detection score — a lone-but-strong detection is kept even if it
     wasn't part of a burst.

Losers are QUARANTINED (moved to captures/rejected/ and hidden from the gallery
and stats), never deleted — the folder is there to spot-check. CPU-free
(pure log + timestamp analysis; no model re-run), runs in the night maintenance
window like the slow-mo verifier.
"""
import re
import json
import logging
import threading
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent
CAPTURES_DIR = BASE / "captures"
REJECTED_DIR = CAPTURES_DIR / "rejected"
THUMBS_DIR = BASE / "thumbnails"
LOG_FILE = BASE / "logs" / "perchsentry.log"
DELETED_FILE = BASE / "deleted.json"

log = logging.getLogger("perchsentry")

# Captures carry an optional detection-score suffix. The live pipeline emits
# "[npu 0.27]"; older lines used "[yolo 0.27]" or omit it entirely. Accept both
# spellings — matching only "yolo" silently dropped the score (group -> None ->
# 0.0), which disabled the strong-single-detection rescue below.
BIRD_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*BIRD DETECTED: .+? \(\d+\.\d+%\) . "
    r"(motion_\S+\.jpg)(?: \[(?:npu|yolo) (\d+\.\d+)\])?"
)


def _load_deleted():
    try:
        return set(json.loads(DELETED_FILE.read_text()))
    except Exception:
        return set()


def _save_deleted(s):
    try:
        DELETED_FILE.write_text(json.dumps(sorted(s)))
    except Exception as e:
        log.warning(f"capture verify: failed to save deleted.json: {e}")


def _parse_bird_captures():
    """[(dt, filename, yolo_score)] for all BIRD DETECTED lines, oldest first."""
    out = []
    if not LOG_FILE.exists():
        return out
    for line in LOG_FILE.read_text(errors="ignore").splitlines():
        m = BIRD_RE.search(line)
        if m:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            fn = m.group(2)
            yolo = float(m.group(3)) if m.group(3) else 0.0
            out.append((dt, fn, yolo))
    out.sort(key=lambda e: e[0])
    return out


def _burst_sizes(entries, gap_secs):
    """filename -> number of captures in its time-cluster."""
    sizes = {}
    cluster = []

    def flush():
        for _, fn, _y in cluster:
            sizes[fn] = len(cluster)

    prev = None
    for e in entries:
        if prev is not None and (e[0] - prev).total_seconds() > gap_secs:
            flush()
            cluster.clear()
        cluster.append(e)
        prev = e[0]
    flush()
    return sizes


class CaptureVerifier:
    def __init__(self, get_settings, clients_active=None, set_maintenance=None):
        self.get_settings = get_settings
        self._clients_active = clients_active or (lambda: False)
        self._set_maintenance = set_maintenance or (lambda *a, **k: None)
        self._stop = threading.Event()
        self._thread = None
        self._running = False

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def run_pass(self, force=False):
        if self._running:
            return 0
        self._running = True
        try:
            s = self.get_settings()
            gap = s.get("capture_verify_gap_secs", 120)
            min_burst = s.get("capture_verify_min_burst", 3)
            strong = s.get("capture_verify_strong_score", 0.35)

            entries = _parse_bird_captures()
            sizes = _burst_sizes(entries, gap)
            REJECTED_DIR.mkdir(exist_ok=True)
            deleted = _load_deleted()

            kept = quarantined = 0
            for dt, fn, yolo in entries:
                if self._stop.is_set():
                    break
                src = CAPTURES_DIR / fn
                if not src.exists():
                    continue  # already gone or previously quarantined
                in_burst = sizes.get(fn, 1) >= min_burst
                strong_single = yolo >= strong
                if in_burst or strong_single:
                    kept += 1
                    continue
                # Quarantine: move out of the gallery and hide from stats.
                try:
                    src.rename(REJECTED_DIR / fn)
                    (THUMBS_DIR / fn).unlink(missing_ok=True)
                    deleted.add(fn)
                    quarantined += 1
                except Exception as e:
                    log.warning(f"capture verify: could not quarantine {fn}: {e}")

            if quarantined:
                _save_deleted(deleted)
            if kept or quarantined:
                log.info(f"Capture verify pass: {kept} kept, {quarantined} quarantined "
                         f"(min_burst={min_burst}, strong>={strong})")
            return quarantined
        finally:
            self._running = False

    def _loop(self):
        self._stop.wait(90)  # settle after startup
        while not self._stop.is_set():
            try:
                s = self.get_settings()
                if s.get("capture_verify_enabled", True):
                    hour = datetime.now().hour
                    start_h = s.get("slowmo_verify_start_hour", 0)
                    end_h = s.get("slowmo_verify_end_hour", 5)
                    if start_h <= hour < end_h and not self._clients_active():
                        self._set_maintenance(True, "Nightly capture review")
                        try:
                            self.run_pass(force=True)
                        finally:
                            self._set_maintenance(False)
            except Exception as e:
                log.error(f"Capture verify loop error: {e}", exc_info=True)
            self._stop.wait(1200)  # re-check every 20 min
