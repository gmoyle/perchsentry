import time
import base64
import logging
import threading
import urllib.request
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

from classify import load_interpreter, load_labels, classify_image
from slowmo import SlowMoCapture, is_hummingbird
from weather import stamp_weather
from objdetect import contains_bird

CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

# Camera AE/AWB is still settling for a moment after (re)start, which can
# produce a huge frame-to-frame diff that looks like motion. Don't trigger
# captures during this window.
STARTUP_GRACE_SECS = 3.0

log = logging.getLogger("birdbuddy")


def notify(species, confidence, settings):
    url = settings.get("ntfy_url", "").strip()
    if not url:
        return
    try:
        headers = {
            "Title": "BirdBuddy",
            "Tags": "bird",
            "Actions": "view, View capture, http://192.168.0.83:8080/, clear=true",
        }
        user = settings.get("ntfy_user", "").strip()
        passwd = settings.get("ntfy_pass", "").strip()
        if user and passwd:
            token = base64.b64encode(f"{user}:{passwd}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        req = urllib.request.Request(
            url,
            data=f"{species} spotted! ({confidence:.1%} confidence)".encode(),
            headers=headers,
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.warning(f"ntfy notification failed: {e}")


def images_are_similar(path_a, path_b, threshold=0.98):
    """Return True if two images are nearly identical (dedup check)."""
    try:
        a = np.array(Image.open(path_a).resize((64, 36))).astype(np.float32)
        b = np.array(Image.open(path_b).resize((64, 36))).astype(np.float32)
        similarity = 1 - np.mean(np.abs(a - b)) / 255
        return similarity >= threshold
    except Exception:
        return False


class MotionDetector:
    def __init__(self, camera, get_settings, clients_active=None):
        self.camera = camera
        self.get_settings = get_settings
        # Returns True when someone is viewing the site; slow-mo is suppressed
        # then to avoid colliding with live-stream/gallery load.
        self._clients_active = clients_active or (lambda: False)
        self._thread = None
        self._stop_event = threading.Event()
        self._interp = load_interpreter()
        self._labels = load_labels()
        self._slowmo = SlowMoCapture(camera)
        self._last_saved_path = None
        self._started_at = 0
        self._status_lock = threading.Lock()
        self._status = {
            "changed_px": 0,
            "min_pixels": 0,
            "motion_crossed": False,
            "last_event": "idle",
            "last_species": None,
            "last_confidence": None,
            "last_event_at": 0,
            "updated_at": 0,
        }

    def _set_status(self, **kwargs):
        with self._status_lock:
            self._status.update(kwargs)
            self._status["updated_at"] = time.time()

    def get_status(self):
        with self._status_lock:
            return dict(self._status)

    def start(self):
        self._stop_event.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _loop(self):
        prev_gray = None
        last_capture = 0
        was_slowmo_active = False

        while not self._stop_event.is_set():
            time.sleep(0.1)
            try:
                slowmo_active = self._slowmo.is_active()
                if was_slowmo_active and not slowmo_active:
                    # Camera just came back from a slow-mo reconfigure — drop
                    # the stale pre-burst frame and re-baseline next tick
                    # instead of diffing against it.
                    prev_gray = None
                was_slowmo_active = slowmo_active

                if time.time() - self._started_at < STARTUP_GRACE_SECS:
                    self.camera.capture_lores()  # drain the stream, discard
                    self._set_status(last_event="warming_up")
                    prev_gray = None
                    continue

                prev_gray, last_capture = self._tick(prev_gray, last_capture)
            except Exception as e:
                # Never let an unexpected error (e.g. the camera briefly
                # dropping its lores stream during a slow-mo burst) kill this
                # thread permanently — log it, reset, and keep going.
                log.error(f"Motion detector loop error: {e}", exc_info=True)
                prev_gray = None

    def _tick(self, prev_gray, last_capture):
        s = self.get_settings()
        gray = self.camera.capture_lores()

        if prev_gray is not None:
            diff = np.abs(gray - prev_gray)
            changed = int(np.sum(diff > s["motion_threshold"]))
            min_pixels = s["motion_min_pixels"]
            crossed = changed > min_pixels
            self._set_status(changed_px=changed, min_pixels=min_pixels, motion_crossed=crossed)

            if crossed:
                now = time.time()
                if now - last_capture > s["motion_cooldown"]:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = CAPTURES_DIR / f"motion_{ts}.jpg"
                    self.camera.capture_file(path)
                    last_capture = now

                    # Deduplication — skip if nearly identical to last save
                    if self._last_saved_path and images_are_similar(path, self._last_saved_path):
                        path.unlink()
                        log.debug(f"Duplicate frame skipped")
                        self._set_status(last_event="duplicate", last_event_at=now)
                        return gray, last_capture
                    self._last_saved_path = path

                    # Object detection pre-filter (fast, skips bird classifier if no animal)
                    if not contains_bird(path):
                        log.debug(f"Pre-filter: no animal detected, deleting {path.name}")
                        path.unlink(missing_ok=True)
                        self._last_saved_path = None
                        self._set_status(last_event="no_animal", last_event_at=now)
                        return gray, last_capture

                    # Weather overlay (stamp before classification so it appears in saved image)
                    if s.get("weather_overlay") and s.get("latitude") and s.get("longitude"):
                        stamp_weather(path, s["latitude"], s["longitude"])

                    result = classify_image(path, self._interp, self._labels)
                    if result["is_bird"]:
                        species = result["species"]
                        confidence = result["confidence"]
                        min_confidence = s.get("confidence_threshold", 30) / 100.0

                        # Always keep hummingbirds regardless of confidence — they're small
                        # and fast so the still is often blurry; better a false-positive slow-mo
                        if confidence < min_confidence and not is_hummingbird(species):
                            log.debug(f"Bird below confidence threshold ({confidence:.1%} < {min_confidence:.1%}), deleting {path.name}")
                            path.unlink(missing_ok=True)
                            self._last_saved_path = None
                            self._set_status(
                                last_event="low_confidence", last_event_at=now,
                                last_species=species, last_confidence=confidence,
                            )
                            return gray, last_capture

                        log.info(f"BIRD DETECTED: {species} ({confidence:.1%}) → {path.name}")
                        self._set_status(
                            last_event="bird_detected", last_event_at=now,
                            last_species=species, last_confidence=confidence,
                        )

                        if not self._slowmo.is_active():
                            if self._clients_active():
                                log.info("Bird detected — skipping slow-mo (someone is viewing the site)")
                            else:
                                log.info("Bird detected! Triggering slow-mo capture")
                                self._slowmo.capture(species, confidence)
                        threading.Thread(
                            target=notify,
                            args=(species, confidence, s),
                            daemon=True,
                        ).start()
                    else:
                        log.debug(f"Motion (no bird/animal, {changed}px), deleting {path.name}")
                        path.unlink(missing_ok=True)
                        self._last_saved_path = None
                        self._set_status(
                            last_event="no_bird", last_event_at=now,
                            last_species=None, last_confidence=result.get("confidence"),
                        )

        return gray, last_capture
