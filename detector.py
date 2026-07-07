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
from slowmo import SlowMoCapture
from daynight import is_daytime
from objdetect import analyze_frame

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
            "Actions": "view, View capture, http://birdbuddy.local:8080/, clear=true",
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


def crop_to_box(path, box, margin=0.2, min_px=48):
    """Save a padded crop of a normalized (x1,y1,x2,y2) box next to `path`.

    Returns the crop path, or None if the box is too small or cropping fails —
    callers then classify the full frame as before.
    """
    try:
        with Image.open(path) as img:
            w, h = img.size
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            x1 = max(0.0, x1 - bw * margin)
            y1 = max(0.0, y1 - bh * margin)
            x2 = min(1.0, x2 + bw * margin)
            y2 = min(1.0, y2 + bh * margin)
            px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
            if px[2] - px[0] < min_px or px[3] - px[1] < min_px:
                return None
            out = path.with_name(path.stem + "_crop.jpg")
            img.crop(px).save(out, "JPEG", quality=90)
            return out
    except Exception:
        return None


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
        self._slowmo = SlowMoCapture(camera, get_settings)
        self._last_saved_path = None
        self._started_at = 0
        self._is_day = True
        self._next_day_check = 0
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
            # 10 Hz motion sampling — fast enough not to miss quick visitors
            # (e.g. hummingbirds) that only dwell a fraction of a second. The
            # encoder-gating change already recovered the idle CPU that a lower
            # rate would have saved, so keep sampling fast.
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

                # No captures at night — this camera has no IR/low-light
                # capability, so night frames are useless. Re-check once a
                # minute; wait in short chunks so shutdown stays responsive.
                s = self.get_settings()
                if s.get("latitude") is not None and s.get("longitude") is not None:
                    now = time.time()
                    if now >= self._next_day_check:
                        self._is_day = is_daytime(s["latitude"], s["longitude"])
                        self._next_day_check = now + 60
                    if not self._is_day:
                        self._set_status(last_event="night_pause")
                        prev_gray = None
                        self._stop_event.wait(30)
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

                    # Object detection pre-filter (NPU): no animal → discard.
                    det = analyze_frame(path)
                    if not det["has_animal"]:
                        log.debug(f"Pre-filter: no animal detected, deleting {path.name}")
                        path.unlink(missing_ok=True)
                        self._last_saved_path = None
                        self._set_status(last_event="no_animal", last_event_at=now)
                        return gray, last_capture

                    # Classify a bird-centered crop when the NPU gave us a box:
                    # the bird fills the classifier input instead of ~5% of the
                    # frame, which dramatically improves species confidence.
                    classify_path = path
                    crop_tmp = None
                    if det.get("box"):
                        crop_tmp = crop_to_box(path, det["box"])
                        if crop_tmp:
                            classify_path = crop_tmp
                    try:
                        result = classify_image(classify_path, self._interp, self._labels)
                    finally:
                        if crop_tmp:
                            crop_tmp.unlink(missing_ok=True)
                    if result["is_bird"]:
                        species = result["species"]
                        confidence = result["confidence"]
                        min_confidence = s.get("confidence_threshold", 30) / 100.0

                        # The confidence slider gates ALL species. (There used
                        # to be a keep-hummingbirds-at-any-confidence bypass
                        # here; it flooded sightings with sub-15% "anna" hits
                        # on empty frames, silently overriding the user's
                        # threshold.)
                        if confidence < min_confidence:
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
                        # Not a bird — but the NPU pre-filter may have labelled
                        # it a known non-bird animal (cat/dog/bear/…). Keep those
                        # as a separate "Animals" track instead of discarding, so
                        # daytime wildlife isn't missed. Bird stats stay pure
                        # because this logs a distinct ANIMAL DETECTED line.
                        animal = det.get("label")
                        if animal and animal != "bird" and s.get("capture_animals", True):
                            score = det.get("score", 0.0)
                            log.info(f"ANIMAL DETECTED: {animal} ({score:.1%}) → {path.name}")
                            self._last_saved_path = path
                            self._set_status(
                                last_event="animal_detected", last_event_at=now,
                                last_species=animal, last_confidence=score,
                            )
                            if not self._slowmo.is_active() and s.get("slowmo_animals", True):
                                if self._clients_active():
                                    log.info(f"{animal} detected — skipping slow-mo (someone is viewing the site)")
                                else:
                                    log.info(f"{animal} detected! Triggering slow-mo capture")
                                    self._slowmo.capture(animal, score)
                            if s.get("notify_animals", False):
                                threading.Thread(
                                    target=notify, args=(animal, score, s), daemon=True,
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
