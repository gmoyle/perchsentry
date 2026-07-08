import time
import base64
import logging
import threading
import subprocess
import urllib.request
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image

from classify import load_interpreter, load_labels, classify_image
import speciesclassify
from slowmo import SlowMoCapture
from daynight import is_daytime
from objdetect import analyze_frame

CAPTURES_DIR = Path(__file__).parent / "captures"
CAPTURES_DIR.mkdir(exist_ok=True)

# Camera AE/AWB is still settling for a moment after (re)start, which can
# produce a huge frame-to-frame diff that looks like motion. Don't trigger
# captures during this window.
STARTUP_GRACE_SECS = 3.0

# Auto-recovery for a wedged Hailo NPU (drops off the PCIe bus). Installed via
# sudoers so the service user may run just this one script as root. Rate-limited
# so a permanently-dead card doesn't reset the bus in a tight loop.
HAILO_RECOVER_CMD = ["sudo", "-n", "/usr/local/sbin/hailo_recover.sh"]
HAILO_RECOVER_COOLDOWN_SECS = 300

log = logging.getLogger("perchsentry")


def notify(species, confidence, settings, message=None):
    url = settings.get("ntfy_url", "").strip()
    if not url:
        return
    try:
        # `message` overrides the default sighting text — used for system
        # alerts (e.g. NPU down) that aren't a species sighting.
        if message is not None:
            body = message
            tags = "warning"
        else:
            body = f"{species} spotted! ({confidence:.1%} confidence)"
            tags = "bird"
        headers = {
            "Title": "PerchSentry",
            "Tags": tags,
            "Actions": "view, View capture, http://perchsentry.local:8080/, clear=true",
        }
        user = settings.get("ntfy_user", "").strip()
        passwd = settings.get("ntfy_pass", "").strip()
        if user and passwd:
            token = base64.b64encode(f"{user}:{passwd}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        req = urllib.request.Request(
            url,
            data=body.encode(),
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
        # SpeciesNet names non-bird animals on the CPU. It's a 214 MB ONNX, so
        # degrade gracefully if it's missing — animals just log generically then.
        try:
            self._species_session = speciesclassify.load_session()
            self._species_labels = speciesclassify.load_labels()
            log.info("SpeciesNet wildlife classifier loaded")
        except Exception as e:
            self._species_session = None
            self._species_labels = None
            log.warning(f"SpeciesNet unavailable ({e}); animals logged generically")
        self._slowmo = SlowMoCapture(camera, get_settings)
        self._last_saved_path = None
        self._started_at = 0
        self._is_day = True
        self._next_day_check = 0
        self._last_npu_recover = 0
        self._status_lock = threading.Lock()
        self._status = {
            "changed_px": 0,
            "min_pixels": 0,
            "motion_crossed": False,
            "last_event": "idle",
            "last_species": None,
            "last_confidence": None,
            "last_filename": None,
            "last_event_at": 0,
            "updated_at": 0,
        }

    def _set_status(self, **kwargs):
        with self._status_lock:
            self._status.update(kwargs)
            self._status["updated_at"] = time.time()

    def get_status(self):
        with self._status_lock:
            st = dict(self._status)
        # Surface NPU pre-filter health so the UI/HA can show when the Hailo
        # card has dropped out (in which case captures are paused, fail-closed).
        try:
            from objdetect import get_health
            st["npu"] = get_health()
        except Exception:
            st["npu"] = None
        return st

    def _name_species(self, path, box, min_conf):
        """Run SpeciesNet on the animal crop MegaDetector found and return
        (common_name, confidence), or None to keep the generic 'animal' label
        (classifier absent, error, low confidence, or a non-species output like
        'blank'/'human')."""
        if self._species_session is None:
            return None
        crop = crop_to_box(path, box) if box else None
        src = crop or path
        try:
            r = speciesclassify.classify_image(src, self._species_session, self._species_labels)
        except Exception as e:
            log.warning(f"Species classify failed: {e}")
            return None
        finally:
            if crop:
                crop.unlink(missing_ok=True)
        if r["confidence"] < min_conf or r["species"] in speciesclassify.NON_SPECIES:
            log.debug(f"SpeciesNet inconclusive: {r['species']} ({r['confidence']:.1%})")
            return None
        return r["species"], r["confidence"]

    def start(self):
        self._stop_event.clear()
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _npu_dead(self):
        try:
            from objdetect import get_health
            return not get_health()["healthy"]
        except Exception:
            return False

    def _try_npu_recover(self):
        """Attempt automatic PCIe recovery of a wedged Hailo card, rate-limited.

        Runs the installed recovery script (remove/rescan + secondary bus reset).
        On success, reset objdetect's health/backend so the next frame re-inits
        and inference resumes. Returns True if the device responded afterwards.
        """
        now = time.time()
        if now - self._last_npu_recover < HAILO_RECOVER_COOLDOWN_SECS:
            return False
        self._last_npu_recover = now
        log.warning("Attempting automatic Hailo PCIe recovery…")
        try:
            proc = subprocess.run(HAILO_RECOVER_CMD, capture_output=True,
                                  text=True, timeout=30)
        except Exception as e:
            log.error(f"Hailo recovery script failed to run: {e}")
            return False
        if proc.returncode == 0:
            log.info("Hailo PCIe recovery succeeded; re-initializing backend")
            try:
                import objdetect
                # Force a clean re-init on the next analyze_frame() and clear the
                # dead flag so the pipeline stops failing closed.
                objdetect._backend = None
                objdetect._npu_dead = False
                objdetect._consecutive_failures = 0
            except Exception:
                pass
            return True
        log.error(f"Hailo PCIe recovery failed (rc={proc.returncode}): "
                  f"{proc.stderr.strip() or proc.stdout.strip()}")
        return False

    def _check_npu_health(self, was_dead):
        """Detect NPU down/up transitions: alert on each, and on going down try
        automatic PCIe recovery. Returns the current dead state to carry over."""
        dead = self._npu_dead()
        if dead and not was_dead:
            recovered = self._try_npu_recover()
            if recovered:
                dead = self._npu_dead()  # may already be healthy again
        if dead != was_dead:
            if dead:
                msg = ("Hailo NPU pre-filter is DOWN — captures paused, "
                       "auto-recovery attempted. Power-cycle may be needed if it "
                       "keeps recurring.")
                log.warning(msg)
            else:
                msg = "Hailo NPU pre-filter recovered — captures resumed."
                log.info(msg)
            threading.Thread(
                target=notify, args=("PerchSentry NPU", None, self.get_settings()),
                kwargs={"message": msg}, daemon=True,
            ).start()
        return dead

    def _loop(self):
        prev_gray = None
        last_capture = 0
        was_slowmo_active = False
        npu_was_dead = False

        while not self._stop_event.is_set():
            # 10 Hz motion sampling — fast enough not to miss quick visitors
            # (e.g. hummingbirds) that only dwell a fraction of a second. The
            # encoder-gating change already recovered the idle CPU that a lower
            # rate would have saved, so keep sampling fast.
            time.sleep(0.1)
            try:
                # Watch NPU pre-filter health: alert on the down/up transitions
                # (captures fail closed while it's down) and attempt automatic
                # PCIe recovery when it dies, so a dropped-out Hailo card self-
                # heals without a manual reset.
                npu_was_dead = self._check_npu_health(npu_was_dead)

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
                            last_filename=path.name,
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
                            # Name the species with SpeciesNet on the same crop;
                            # fall back to the generic "animal" label if it's
                            # unsure. This confidence is the classifier's, which
                            # is a different (and more meaningful) number than
                            # the detector's box score it replaces.
                            named = self._name_species(
                                path, det.get("box"),
                                s.get("species_confidence", 50) / 100.0,
                            )
                            if named:
                                animal, score = named
                            log.info(f"ANIMAL DETECTED: {animal} ({score:.1%}) → {path.name}")
                            self._last_saved_path = path
                            self._set_status(
                                last_event="animal_detected", last_event_at=now,
                                last_species=animal, last_confidence=score,
                                last_filename=path.name,
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
