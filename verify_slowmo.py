"""
Nightly slow-mo verifier.

Slow-mo bursts fire on any bird detection, so the folder fills with false
triggers (wind, chains, low-confidence misclassifications). This samples a few
frames from each video, runs the TFLite species classifier (the Hailo NPU path
is currently non-functional), and quarantines any video where no confident
hummingbird appears — moving it to slowmo/rejected/ rather than deleting, so
nothing is lost.

Deliberately gentle: single-threaded TFLite + ffmpeg and a throttle between
videos, so it doesn't spike all cores (this board has hard-hung under sustained
load). Runs only after midnight, and puts the site into maintenance mode for
the duration so no browser traffic competes with it.
"""
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

from classify import load_interpreter, load_labels, classify_image
from slowmo import is_hummingbird, SLOWMO_DIR

REJECTED_DIR = SLOWMO_DIR / "rejected"
log = logging.getLogger("birdbuddy")

# Fractions of the clip to sample: first, quarter, middle, three-quarter, last.
SAMPLE_FRACTIONS = [0.02, 0.25, 0.5, 0.75, 0.98]


def sidecar_path(video):
    return video.with_suffix(".json")


def load_meta(video):
    p = sidecar_path(video)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_meta(video, meta):
    try:
        sidecar_path(video).write_text(json.dumps(meta))
    except Exception as e:
        log.warning(f"Slow-mo sidecar save failed for {video.name}: {e}")


def _video_duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def _extract_frame(video, t, out):
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-threads", "1", "-ss", f"{t:.3f}", "-i", str(video),
             "-frames:v", "1", "-q:v", "3", str(out)],
            capture_output=True, timeout=30,
        )
        return out.exists() and out.stat().st_size > 0
    except Exception:
        return False


def verify_video(video, interp, labels, min_conf):
    """Return (is_hummingbird_found, best_result_dict)."""
    dur = _video_duration(video)
    best = {"species": None, "confidence": 0.0}
    tmp = SLOWMO_DIR / ("_verify_" + video.stem + ".jpg")
    found = False
    try:
        for frac in SAMPLE_FRACTIONS:
            t = max(0.0, dur * frac)
            if not _extract_frame(video, t, tmp):
                continue
            try:
                r = classify_image(tmp, interp, labels)
            except Exception:
                r = None
            if r and r.get("species"):
                if r["confidence"] > best["confidence"]:
                    best = {"species": r["species"], "confidence": r["confidence"]}
                if is_hummingbird(r["species"]) and r["confidence"] >= min_conf:
                    found = True
                    break
    finally:
        tmp.unlink(missing_ok=True)
    return found, best


class SlowMoVerifier:
    def __init__(self, get_settings, clients_active=None, set_maintenance=None):
        self.get_settings = get_settings
        self._clients_active = clients_active or (lambda: False)
        self._set_maintenance = set_maintenance or (lambda *a, **k: None)
        self._interp = None
        self._labels = None
        self._stop = threading.Event()
        self._thread = None
        self._running = False

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _ensure_model(self):
        if self._interp is None:
            # Single-threaded to keep load off the other cores.
            self._interp = load_interpreter(num_threads=1)
            self._labels = load_labels()

    def _pending(self):
        """Slow-mo videos not yet verified."""
        out = []
        for v in sorted(SLOWMO_DIR.glob("slowmo_*.mp4")):
            if load_meta(v).get("verified") is None:
                out.append(v)
        return out

    def has_pending(self):
        return bool(self._pending())

    def run_pass(self, force=False):
        """Verify all not-yet-checked slow-mo videos. Returns count processed."""
        if self._running:
            return 0
        self._running = True
        try:
            self._ensure_model()
            REJECTED_DIR.mkdir(exist_ok=True)
            s = self.get_settings()
            min_conf = s.get("slowmo_verify_confidence", 0.25)
            throttle = s.get("slowmo_verify_throttle", 3.0)
            processed = kept = quarantined = 0
            for v in self._pending():
                if self._stop.is_set():
                    break
                if not force and self._clients_active():
                    log.info("Slow-mo verify: paused (someone is viewing the site)")
                    break
                found, best = verify_video(v, self._interp, self._labels, min_conf)
                meta = load_meta(v)
                meta.update({
                    "verified": bool(found),
                    "verified_at": datetime.now().isoformat(timespec="seconds"),
                    "best_species": best["species"],
                    "best_confidence": round(best["confidence"], 4),
                })
                save_meta(v, meta)
                if found:
                    kept += 1
                    log.info(f"Slow-mo verify: KEEP {v.name} "
                             f"({best['species']} {best['confidence']:.0%})")
                else:
                    dest = REJECTED_DIR / v.name
                    v.rename(dest)
                    sc = sidecar_path(v)
                    if sc.exists():
                        sc.rename(REJECTED_DIR / sc.name)
                    quarantined += 1
                    bs = best["species"] or "nothing"
                    log.info(f"Slow-mo verify: QUARANTINE {v.name} "
                             f"(best: {bs} {best['confidence']:.0%})")
                processed += 1
                self._stop.wait(throttle)  # gentle throttle between videos
            if processed:
                log.info(f"Slow-mo verify pass: {processed} checked, "
                         f"{kept} kept, {quarantined} quarantined")
            return processed
        finally:
            self._running = False

    def _loop(self):
        self._stop.wait(60)  # settle after startup
        while not self._stop.is_set():
            try:
                s = self.get_settings()
                if s.get("slowmo_verify_enabled", True):
                    hour = datetime.now().hour
                    start_h = s.get("slowmo_verify_start_hour", 0)   # after midnight
                    end_h = s.get("slowmo_verify_end_hour", 5)
                    in_window = start_h <= hour < end_h
                    if in_window and not self._clients_active() and self.has_pending():
                        self._set_maintenance(True, "Nightly slow-mo verification")
                        try:
                            self.run_pass(force=True)
                        finally:
                            self._set_maintenance(False)
            except Exception as e:
                log.error(f"Slow-mo verify loop error: {e}", exc_info=True)
            self._stop.wait(600)  # re-check every 10 minutes
