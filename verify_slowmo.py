"""
Nightly slow-mo verifier.

Slow-mo bursts fire on any bird detection, so the folder fills with false
triggers (wind, chains, low-confidence misclassifications). This samples a few
frames from each video and keeps the clip if the NPU finds an animal bounding
box in any sampled frame (falling back to the TFLite species classifier when
the NPU isn't available). Rejects are quarantined to slowmo/rejected/ — moved,
not deleted, so nothing is lost.

For kept clips it also:
- picks the best frame (largest, highest-scoring bird box) and saves it as a
  poster JPEG next to the video for gallery thumbnails, and
- labels the species by classifying a bird-centered crop of that frame.

Deliberately gentle: single-threaded TFLite + ffmpeg and a throttle between
videos, so it doesn't spike all cores (this board has hard-hung under sustained
load). Runs only after midnight, and puts the site into maintenance mode for
the duration so no browser traffic competes with it.
"""
import json
import shutil
import logging
import subprocess
import threading
from pathlib import Path
from datetime import datetime

from classify import load_interpreter, load_labels, classify_image
from slowmo import SLOWMO_DIR
from objdetect import analyze_frame
from detector import crop_to_box

REJECTED_DIR = SLOWMO_DIR / "rejected"
log = logging.getLogger("birdbuddy")

# Fractions of the clip to sample: first, quarter, middle, three-quarter, last.
SAMPLE_FRACTIONS = [0.02, 0.25, 0.5, 0.75, 0.98]


def sidecar_path(video):
    return video.with_suffix(".json")


def poster_path(video):
    return video.with_name(video.stem + "_poster.jpg")


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


def _box_area(box):
    if not box:
        return 0.0
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def verify_video(video, interp, labels, min_conf):
    """Sample frames and judge the clip.

    Returns (bird_found, best) where best = {species, confidence, det_label,
    det_score, frame_time}. When a bird is found, the best sampled frame is
    saved as a poster JPEG next to the video.
    """
    dur = _video_duration(video)
    best = {"species": None, "confidence": 0.0, "det_label": None,
            "det_score": 0.0, "frame_time": None}
    best_rank = -1.0
    tmp = SLOWMO_DIR / ("_verify_" + video.stem + ".jpg")
    best_tmp = SLOWMO_DIR / ("_verify_best_" + video.stem + ".jpg")
    found = False
    try:
        for frac in SAMPLE_FRACTIONS:
            t = max(0.0, dur * frac)
            if not _extract_frame(video, t, tmp):
                continue

            det = analyze_frame(tmp)

            # Species: classify a bird-centered crop when we have a box —
            # far more reliable than classifying the whole frame.
            r = None
            classify_src = tmp
            crop = crop_to_box(tmp, det["box"]) if det.get("box") else None
            if crop:
                classify_src = crop
            try:
                r = classify_image(classify_src, interp, labels)
            except Exception:
                r = None
            finally:
                if crop:
                    crop.unlink(missing_ok=True)

            frame_found = False
            if det["supported"]:
                # NPU verdict: an animal box in frame keeps the clip.
                frame_found = det["has_animal"] and det.get("box") is not None
            elif r and r.get("is_bird") and r["confidence"] >= min_conf:
                # Fallback (no NPU): confident bird from the classifier.
                frame_found = True

            # Rank frames: prefer bird boxes, larger + higher-scoring wins.
            rank = 0.0
            if det.get("box"):
                rank = _box_area(det["box"]) * max(det["score"], 0.01)
                if det.get("label") == "bird":
                    rank *= 2.0
            elif r and r.get("is_bird"):
                rank = r["confidence"] * 0.001  # weak, but better than nothing

            if rank > best_rank:
                best_rank = rank
                best = {
                    "species": (r["species"] if r and r.get("is_bird") else None),
                    "confidence": (r["confidence"] if r and r.get("is_bird") else 0.0),
                    "det_label": det.get("label"),
                    "det_score": round(det.get("score", 0.0), 4),
                    "frame_time": round(t, 3),
                }
                shutil.copyfile(tmp, best_tmp)

            if frame_found:
                found = True
                # Don't break: later frames may rank better for the poster.

        if found and best_tmp.exists():
            shutil.move(str(best_tmp), str(poster_path(video)))
    finally:
        tmp.unlink(missing_ok=True)
        best_tmp.unlink(missing_ok=True)
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
                    "det_label": best["det_label"],
                    "det_score": best["det_score"],
                })
                save_meta(v, meta)
                if found:
                    kept += 1
                    label = best["species"] or best["det_label"] or "bird"
                    log.info(f"Slow-mo verify: KEEP {v.name} "
                             f"({label} {best['confidence']:.0%})")
                else:
                    for src in (v, sidecar_path(v), poster_path(v)):
                        if src.exists():
                            src.rename(REJECTED_DIR / src.name)
                    quarantined += 1
                    bs = best["species"] or best["det_label"] or "nothing"
                    log.info(f"Slow-mo verify: QUARANTINE {v.name} (best: {bs})")
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
